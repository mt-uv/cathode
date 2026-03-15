import tempfile
from ase.io.trajectory import Trajectory

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter

import io
import math
import os
import random
import tempfile
from functools import lru_cache
from typing import Callable, Dict, List

import numpy as np

from ase.io import read
from ase.build import bulk
from ase.optimize import LBFGS, BFGS, FIRE
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from huggingface_hub import hf_hub_download

from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

from torch.serialization import add_safe_globals

# ONLY do this if you trust the checkpoint source (official fairchem weights)
add_safe_globals([slice])

TEMPLATE_PATH = "../template/Na12Mn11Zr1O24.cif"

# Fixed workflow assumptions
NA_REMOVED_FIXED = 12
RELAX_FMAX = 0.05
RELAX_STEPS = 300
DEVICE = "cpu"
N_CONFIGS = 10

# MD settings
MD_TEMP_K = 800
MD_TIMESTEP_FS = 1.0
MD_STEPS = 5000
MD_SAMPLE_INTERVAL = 1
MD_NA_VACANCY_FRACTION = 0.25
MD_LOG_INTERVAL = 100

SUPPORTED_POTENTIALS = {"uma", "orb"}

ATOMIC_WEIGHTS = {
    "Na": 22.98976928,
    "C": 12.011,
    "O": 15.999,
    "Mn": 54.938044,
    "Ni": 58.6934,
    "Co": 58.933194,
    "Fe": 55.845,
    "Cr": 51.9961,
    "V": 50.9415,
    "Ti": 47.867,
    "Mg": 24.305,
    "Al": 26.9815385,
    "Zn": 65.38,
    "Cu": 63.546,
    "Zr": 91.224,
    "Y": 88.90584,
    "Nb": 92.90637,
}

OXIDE_PRECURSORS = {
    "Mn": {"formula": "Mn2O3", "metal_per_formula": 2},
    "Ni": {"formula": "NiO", "metal_per_formula": 1},
    "Co": {"formula": "Co3O4", "metal_per_formula": 3},
    "Fe": {"formula": "Fe2O3", "metal_per_formula": 2},
    "Cr": {"formula": "Cr2O3", "metal_per_formula": 2},
    "V": {"formula": "V2O5", "metal_per_formula": 2},
    "Ti": {"formula": "TiO2", "metal_per_formula": 1},
    "Mg": {"formula": "MgO", "metal_per_formula": 1},
    "Al": {"formula": "Al2O3", "metal_per_formula": 2},
    "Zn": {"formula": "ZnO", "metal_per_formula": 1},
    "Cu": {"formula": "CuO", "metal_per_formula": 1},
    "Zr": {"formula": "ZrO2", "metal_per_formula": 1},
    "Y": {"formula": "Y2O3", "metal_per_formula": 2},
    "Nb": {"formula": "Nb2O5", "metal_per_formula": 2},
}

NA_PRECURSOR = {"formula": "Na2CO3", "metal_per_formula": 2}


def log(msg: str):
    print(msg, flush=True)


def atoms_to_cif_string(atoms) -> str:
    struct = AseAtomsAdaptor.get_structure(atoms)
    return str(CifWriter(struct))


def cif_string_to_atoms(cif_text: str):
    if not cif_text or not cif_text.strip():
        raise ValueError("CIF input is empty")
    return read(io.StringIO(cif_text), format="cif")

def uploaded_file_to_atoms(filename: str, file_bytes: bytes):
    suffix = os.path.splitext(filename)[1] or ".tmp"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        atoms = read(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return atoms


def get_optimizer(name: str, atoms, logfile="-"):
    key = (name or "LBFGS").upper()

    if key == "LBFGS":
        return LBFGS(atoms, logfile=logfile)
    if key == "BFGS":
        return BFGS(atoms, logfile=logfile)
    if key == "FIRE":
        return FIRE(atoms, logfile=logfile)

    raise ValueError(f"Unsupported optimizer '{name}'")


@lru_cache(maxsize=1)
def get_uma_calc():
    log("[INIT] Loading UMA calculator...")
    checkpoint = hf_hub_download(
        repo_id="facebook/UMA",
        filename="uma-s-1p1.pt",
        subfolder="checkpoints",
    )

    predictor = load_predict_unit(
        checkpoint,
        inference_settings="default",
        device=DEVICE,
    )

    log("[INIT] UMA calculator ready")
    return FAIRChemCalculator(predictor, task_name="omat")


@lru_cache(maxsize=1)
def get_orb_calc():
    log("[INIT] Loading ORB calculator...")
    orbff = pretrained.orb_v3_conservative_inf_omat(
        device=DEVICE,
        precision="float32-high",
    )
    log("[INIT] ORB calculator ready")
    return ORBCalculator(orbff, device=DEVICE)


def normalize_potential(potential: str) -> str:
    potential = (potential or "uma").lower().strip()
    if potential not in SUPPORTED_POTENTIALS:
        raise ValueError(
            f"Unsupported potential '{potential}'. Choose one of: {sorted(SUPPORTED_POTENTIALS)}"
        )
    return potential

def uploaded_file_to_atoms(filename: str, file_bytes: bytes):
    suffix = os.path.splitext(filename)[1] or ".tmp"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        atoms = read(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return atoms

def run_relaxation_stream(filename: str, file_bytes: bytes, potential="uma", fmax=0.05, steps=300):
    potential = normalize_potential(potential)
    calc = get_calc(potential)

    yield {
        "event": "status",
        "message": f"Loading uploaded structure: {filename}",
    }

    atoms = uploaded_file_to_atoms(filename, file_bytes)

    yield {
        "event": "status",
        "message": "Structure loaded",
        "n_atoms": len(atoms),
        "progress": 0.1,
    }

    atoms.calc = calc
    initial_energy = float(atoms.get_potential_energy())

    yield {
        "event": "status",
        "message": "Starting relaxation",
        "initial_energy": initial_energy,
        "progress": 0.2,
    }

    final_energy = relax_ase(
        atoms,
        calc=calc,
        fmax=fmax,
        steps=steps,
        label=f"uploaded relaxation ({filename})",
    )

    relaxed_cif = atoms_to_cif_string(atoms)

    yield {
        "event": "result",
        "filename": filename,
        "potential": potential,
        "n_atoms": len(atoms),
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "relaxed_cif": relaxed_cif,
        "progress": 1.0,
    }

def run_uploaded_md_stream(
    filename: str,
    file_bytes: bytes,
    potential="uma",
    temperature_k=800.0,
    timestep_fs=1.0,
    total_time_ps=5.0,
    should_stop: Callable[[], bool] | None = None,
):
    potential = normalize_potential(potential)
    should_stop = should_stop or (lambda: False)

    atoms = uploaded_file_to_atoms(filename, file_bytes)
    calc = get_calc(potential)

    atoms.calc = calc

    steps = max(1, int(round(float(total_time_ps) * 1000.0 / float(timestep_fs))))

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k)

    dyn = Langevin(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_k,
        friction=md_friction_for_potential(potential),
    )

    pbc = atoms.get_pbc()
    cell = np.array(atoms.get_cell())

    ref_scaled = atoms.get_scaled_positions(wrap=True)
    prev_scaled = ref_scaled.copy()
    cumulative_frac = np.zeros_like(ref_scaled)

    temp_series_k = []

    initial_cif = atoms_to_cif_string(atoms)
    species_symbols = [a.symbol for a in atoms]
    unique_species = sorted(set(species_symbols))

    traj_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".traj")
    traj_tmp.close()
    traj_path = traj_tmp.name
    traj = Trajectory(traj_path, "w", atoms)

    yield {
        "event": "meta",
        "filename": filename,
        "potential": potential,
        "temperature_k": float(temperature_k),
        "timestep_fs": float(timestep_fs),
        "total_time_ps": float(total_time_ps),
        "steps": steps,
        "n_atoms": len(atoms),
        "initial_cif": initial_cif,
        "species": unique_species,
    }

    for step in range(1, steps + 1):
        if should_stop():
            avg_temp = float(np.mean(temp_series_k)) if temp_series_k else float(temperature_k)
            traj.close()
            yield {
                "event": "cancelled",
                "message": f"Upload MD stopped by user at step {step}",
                "avg_temperature_k": avg_temp,
                "step": step,
                "steps": steps,
            }
            return

        dyn.run(1)
        traj.write(atoms)

        curr_scaled = atoms.get_scaled_positions(wrap=True)
        delta = curr_scaled - prev_scaled

        delta[:, 0] = delta[:, 0] - np.round(delta[:, 0]) if pbc[0] else delta[:, 0]
        delta[:, 1] = delta[:, 1] - np.round(delta[:, 1]) if pbc[1] else delta[:, 1]
        delta[:, 2] = delta[:, 2] - np.round(delta[:, 2]) if pbc[2] else delta[:, 2]

        cumulative_frac += delta
        prev_scaled = curr_scaled

        disp_cart = cumulative_frac @ cell
        sq = np.sum(disp_cart ** 2, axis=1)

        time_ps = step * float(timestep_fs) / 1000.0

        msd_by_species = {}
        for sp in unique_species:
            mask = np.array([a.symbol == sp for a in atoms], dtype=bool)
            msd_by_species[sp] = float(np.mean(sq[mask])) if np.any(mask) else 0.0

        ek = atoms.get_kinetic_energy()
        t_inst = ek / (1.5 * units.kB * len(atoms))
        temp_series_k.append(float(t_inst))

        yield {
            "event": "progress",
            "step": step,
            "steps": steps,
            "time_ps": float(time_ps),
            "msd_by_species": msd_by_species,
            "temperature_k": float(t_inst),
            "progress": float(step / steps),
        }

    traj.close()

    final_cif = atoms_to_cif_string(atoms)

    yield {
        "event": "result",
        "filename": filename,
        "potential": potential,
        "temperature_k": float(temperature_k),
        "timestep_fs": float(timestep_fs),
        "total_time_ps": float(total_time_ps),
        "steps": steps,
        "avg_temperature_k": float(np.mean(temp_series_k)) if temp_series_k else float(temperature_k),
        "final_temperature_k": float(temp_series_k[-1]) if temp_series_k else float(temperature_k),
        "final_cif": final_cif,
        "traj_path": traj_path,
    }



def get_calc(potential: str):
    potential = normalize_potential(potential)

    if potential == "uma":
        return get_uma_calc()
    if potential == "orb":
        return get_orb_calc()

    raise ValueError(f"Unsupported potential '{potential}'")


def md_friction_for_potential(potential: str) -> float:
    return 0.02 if potential == "orb" else 0.01


def relax_ase(atoms, calc, fmax=RELAX_FMAX, steps=RELAX_STEPS, label="relaxation") -> float:
    log(f"[RELAX] Starting {label} (fmax={fmax}, steps={steps})")
    atoms.calc = calc
    dyn = LBFGS(atoms, logfile="-")
    dyn.run(fmax=fmax, steps=steps)
    energy = float(atoms.get_potential_energy())
    log(f"[RELAX] Finished {label} | Energy = {energy:.6f} eV")
    return energy


def compute_voltage(E_sod, E_desod, mu_na, n_removed=NA_REMOVED_FIXED) -> float:
    return (E_desod - E_sod + n_removed * mu_na) / n_removed


@lru_cache(maxsize=None)
def get_mu_na(potential: str) -> float:
    potential = normalize_potential(potential)
    log(f"[REF] Computing Na chemical potential with {potential.upper()}...")
    calc = get_calc(potential)

    na_bulk = bulk("Na", "bcc", a=4.23)
    na_bulk.calc = calc
    E = float(na_bulk.get_potential_energy())
    mu = E / len(na_bulk)
    log(f"[REF] mu_Na = {mu:.6f} eV/atom")
    return mu


def validate_fractions(transition_metals, dopants, fractions):
    allowed = set(transition_metals) | set(dopants)

    if not fractions:
        raise ValueError("fractions dictionary is empty")

    fraction_keys = set(fractions.keys())

    if fraction_keys != allowed:
        missing = sorted(allowed - fraction_keys)
        extra = sorted(fraction_keys - allowed)
        pieces = []
        if missing:
            pieces.append(f"missing keys: {missing}")
        if extra:
            pieces.append(f"unexpected keys: {extra}")
        raise ValueError(
            "fractions keys must match selected TM + dopants exactly; " + ", ".join(pieces)
        )

    total = 0.0
    positive_count = 0

    for el, value in fractions.items():
        try:
            v = float(value)
        except Exception:
            raise ValueError(f"Fraction for {el} is not a valid number")

        if v < 0 or v > 1:
            raise ValueError(f"Fraction for {el} must be between 0 and 1")

        if v > 0:
            positive_count += 1

        total += v

    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"TM + dopant fractions must sum to 1. Current sum = {total:.6f}")

    if positive_count == 0:
        raise ValueError("At least one fraction must be > 0")


def fractions_to_counts(fractions, total_sites: int):
    items = list(fractions.items())

    raw = [(el, float(frac) * total_sites) for el, frac in items]
    base = [(el, int(val)) for el, val in raw]

    counts = {el: c for el, c in base}
    used = sum(counts.values())
    remainder = total_sites - used

    remainders = sorted(
        [(el, raw_val - int(raw_val)) for el, raw_val in raw],
        key=lambda x: x[1],
        reverse=True,
    )

    for i in range(remainder):
        el = remainders[i][0]
        counts[el] += 1

    return counts


def counts_to_species_list(counts):
    species = []
    for el, count in counts.items():
        species.extend([el] * int(count))
    return species


def generate_unique_config_signatures(counts, n_configs=10, max_attempts=2000):
    log(f"[CONFIG] Generating up to {n_configs} unique configurations...")
    base_species = counts_to_species_list(counts)

    if len(base_species) == 0:
        raise ValueError("No species generated from counts")

    seen = set()
    signatures = []

    attempts = 0
    while len(signatures) < n_configs and attempts < max_attempts:
        attempts += 1
        trial = base_species.copy()
        random.shuffle(trial)
        key = tuple(trial)
        if key in seen:
            continue
        seen.add(key)
        signatures.append(trial)

    if len(signatures) == 0:
        raise ValueError("Could not generate any configuration signatures")

    log(f"[CONFIG] Generated {len(signatures)} unique configurations")
    return signatures


def apply_signature_to_atoms(template_atoms, tm_indices, signature):
    atoms = template_atoms.copy()
    for idx, symbol in zip(tm_indices, signature):
        atoms[idx].symbol = symbol
    return atoms


def summarize_species(counts, transition_metals, dopants):
    tm_species = {el: int(counts[el]) for el in transition_metals if counts.get(el, 0) > 0}
    dopant_species = {el: int(counts[el]) for el in dopants if counts.get(el, 0) > 0}

    chosen_tm = ", ".join(f"{el}{tm_species[el]}" for el in tm_species) if tm_species else "—"
    chosen_dopant = ", ".join(f"{el}{dopant_species[el]}" for el in dopant_species) if dopant_species else "—"

    return tm_species, dopant_species, chosen_tm, chosen_dopant


def create_random_na_vacancies(atoms, vacancy_fraction=MD_NA_VACANCY_FRACTION):
    atoms_vac = atoms.copy()
    na_indices = [i for i, a in enumerate(atoms_vac) if a.symbol == "Na"]

    if len(na_indices) == 0:
        raise ValueError("No Na atoms found for vacancy creation")

    n_remove = max(1, int(round(vacancy_fraction * len(na_indices))))
    remove_indices = random.sample(na_indices, n_remove)

    log(
        f"[MD] Creating random Na vacancies: removing {n_remove}/{len(na_indices)} "
        f"Na atoms ({100.0 * vacancy_fraction:.1f}%)"
    )

    for idx in sorted(remove_indices, reverse=True):
        del atoms_vac[idx]

    return atoms_vac, n_remove


def _format_float(x: float, ndigits: int = 4) -> str:
    s = f"{x:.{ndigits}f}"
    s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def _formula_from_composition(composition: Dict[str, float]) -> str:
    pieces = []
    for el, frac in composition.items():
        if float(frac) > 0:
            pieces.append(f"{el}{_format_float(float(frac), 3)}")
    return "Na1 " + " ".join(pieces) + " O2"


def _parse_formula(formula: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    i = 0
    while i < len(formula):
        if not formula[i].isalpha() or not formula[i].isupper():
            raise ValueError(f"Invalid formula: {formula}")
        j = i + 1
        while j < len(formula) and formula[j].islower():
            j += 1
        el = formula[i:j]
        k = j
        while k < len(formula) and formula[k].isdigit():
            k += 1
        count = int(formula[j:k]) if k > j else 1
        out[el] = out.get(el, 0) + count
        i = k
    return out


def _molar_mass(formula: str) -> float:
    parsed = _parse_formula(formula)
    total = 0.0
    for el, count in parsed.items():
        if el not in ATOMIC_WEIGHTS:
            raise ValueError(f"Missing atomic weight for {el}")
        total += ATOMIC_WEIGHTS[el] * count
    return total


def generate_solid_state_synthesis_route(
    composition: Dict[str, float],
    batch_mmol: float = 10.0,
    na_excess_fraction: float = 0.05,
):
    if not composition:
        raise ValueError("composition is empty")

    cleaned: Dict[str, float] = {}
    total = 0.0
    for el, frac in composition.items():
        v = float(frac)
        if v < 0:
            raise ValueError(f"Negative fraction for {el}")
        if v > 0:
            cleaned[el] = v
            total += v

    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Composition fractions must sum to 1. Current sum = {total:.6f}")

    unsupported = [el for el in cleaned if el not in OXIDE_PRECURSORS]
    if unsupported:
        raise ValueError(f"No oxide precursor mapping defined for: {unsupported}")

    if batch_mmol <= 0:
        raise ValueError("batch_mmol must be > 0")

    if na_excess_fraction < 0 or na_excess_fraction > 0.5:
        raise ValueError("na_excess_fraction must be between 0 and 0.5")

    batch_mol = batch_mmol / 1000.0
    formula = _formula_from_composition(cleaned)

    precursors: List[Dict] = []

    na2co3_formula = NA_PRECURSOR["formula"]
    na2co3_metal_per_formula = NA_PRECURSOR["metal_per_formula"]
    na_moles_needed = batch_mol * 1.0 * (1.0 + na_excess_fraction)
    na2co3_moles = na_moles_needed / na2co3_metal_per_formula
    na2co3_mm = _molar_mass(na2co3_formula)
    precursors.append({
        "element": "Na",
        "fraction": 1.0,
        "precursor": na2co3_formula,
        "metal_per_precursor": na2co3_metal_per_formula,
        "moles_precursor": na2co3_moles,
        "mmol_precursor": na2co3_moles * 1000.0,
        "molar_mass_g_mol": na2co3_mm,
        "mass_g": na2co3_moles * na2co3_mm,
        "note": f"{int(round(na_excess_fraction * 100))}% Na excess",
    })

    for el, frac in cleaned.items():
        precursor_info = OXIDE_PRECURSORS[el]
        precursor_formula = precursor_info["formula"]
        metal_per_formula = precursor_info["metal_per_formula"]
        moles_precursor = batch_mol * frac / metal_per_formula
        mm = _molar_mass(precursor_formula)

        precursors.append({
            "element": el,
            "fraction": float(frac),
            "precursor": precursor_formula,
            "metal_per_precursor": metal_per_formula,
            "moles_precursor": moles_precursor,
            "mmol_precursor": moles_precursor * 1000.0,
            "molar_mass_g_mol": mm,
            "mass_g": moles_precursor * mm,
            "note": "oxide precursor",
        })

    precursor_lines = []
    for p in precursors:
        precursor_lines.append(
            f"- {p['precursor']}: {_format_float(p['mmol_precursor'], 4)} mmol "
            f"({_format_float(p['mass_g'], 4)} g)"
        )

    procedure = f"""Target composition: {formula}

Recommended solid-state synthesis route:
1. Dry all starting powders (especially Na2CO3) at 120 °C for 6-12 h before weighing.
2. Weigh the following precursors for a {batch_mmol:.2f} mmol target batch:
{chr(10).join(precursor_lines)}
3. Mix thoroughly in an agate mortar or ball mill for 30-60 min using oxide precursors and Na2CO3 as the sodium source.
4. Pre-calcine the mixed powder in air at 500 °C for 4-6 h to decompose Na2CO3 and homogenize the mixture.
5. Regrind the powder thoroughly, then press into pellets.
6. Calcine/sinter in air (or flowing O2 if preferred) at 850-900 °C for 10-15 h in an alumina crucible with a loose-fitting lid.
7. Cool, regrind, repelletize, and perform a second sintering step at 850-900 °C for another 10-15 h for better phase formation.
8. Cool to room temperature naturally unless a quench is specifically desired for phase retention studies.
9. Store the final powder in a dry container.

Typical practical notes:
- Use ~{int(round(na_excess_fraction * 100))}% excess Na to compensate for sodium loss at high temperature.
- A covered alumina crucible is recommended to reduce Na evaporation.
- Final optimization of temperature/time may be needed depending on the exact TM/dopant combination and phase purity.
- PXRD after each firing step is recommended to confirm formation of the layered phase.
"""

    return {
        "formula": formula,
        "batch_mmol": float(batch_mmol),
        "na_excess_fraction": float(na_excess_fraction),
        "precursors": precursors,
        "procedure": procedure,
    }


def run_md_stream(cif: str, potential="uma", should_stop: Callable[[], bool] | None = None):
    potential = normalize_potential(potential)
    log(f"[MD] Starting streaming MD workflow with {potential.upper()}")

    should_stop = should_stop or (lambda: False)

    def cancelled_payload(message: str, **extra):
        return {
            "event": "cancelled",
            "message": message,
            "potential": potential,
            **extra,
        }

    calc = get_calc(potential)

    if should_stop():
        yield cancelled_payload("MD cancelled before structure loading")
        return

    atoms = cif_string_to_atoms(cif)

    yield {
        "event": "status",
        "message": "Loaded selected structure for MD",
        "potential": potential,
    }

    if should_stop():
        yield cancelled_payload("MD cancelled before vacancy creation")
        return

    atoms_md_seed, md_na_removed = create_random_na_vacancies(
        atoms,
        vacancy_fraction=MD_NA_VACANCY_FRACTION,
    )

    yield {
        "event": "status",
        "message": "Created random 25% Na vacancies",
        "na_removed_for_md": int(md_na_removed),
        "na_vacancy_fraction": MD_NA_VACANCY_FRACTION,
    }

    atoms_md = atoms_md_seed.copy()
    atoms_md.calc = calc

    if should_stop():
        yield cancelled_payload(
            "MD cancelled before pre-relaxation",
            na_removed_for_md=int(md_na_removed),
            na_vacancy_fraction=MD_NA_VACANCY_FRACTION,
        )
        return

    yield {
        "event": "status",
        "message": "Starting MD pre-relaxation",
    }

    relax_ase(atoms_md, calc=calc, fmax=0.05, steps=120, label="MD pre-relaxation")

    cif_md_start = atoms_to_cif_string(atoms_md)

    yield {
        "event": "status",
        "message": "MD pre-relaxation finished",
        "cif_md_start": cif_md_start,
    }

    if should_stop():
        yield cancelled_payload(
            "MD cancelled after pre-relaxation",
            cif_md_start=cif_md_start,
            na_removed_for_md=int(md_na_removed),
            na_vacancy_fraction=MD_NA_VACANCY_FRACTION,
        )
        return

    MaxwellBoltzmannDistribution(atoms_md, temperature_K=MD_TEMP_K)

    dyn = Langevin(
        atoms_md,
        timestep=MD_TIMESTEP_FS * units.fs,
        temperature_K=MD_TEMP_K,
        friction=md_friction_for_potential(potential),
    )

    pbc = atoms_md.get_pbc()
    cell = np.array(atoms_md.get_cell())

    ref_scaled = atoms_md.get_scaled_positions(wrap=True)
    prev_scaled = ref_scaled.copy()
    cumulative_frac = np.zeros_like(ref_scaled)

    na_mask = np.array([a.symbol == "Na" for a in atoms_md], dtype=bool)
    non_na_mask = ~na_mask

    temp_series_k = []

    yield {
        "event": "meta",
        "potential": potential,
        "temperature_k": MD_TEMP_K,
        "timestep_fs": MD_TIMESTEP_FS,
        "steps": MD_STEPS,
        "sample_interval": MD_SAMPLE_INTERVAL,
        "total_time_ps": MD_STEPS * MD_TIMESTEP_FS / 1000.0,
        "n_atoms": len(atoms_md),
        "n_na_atoms": int(np.sum(na_mask)),
        "n_non_na_atoms": int(np.sum(non_na_mask)),
        "na_vacancy_fraction": MD_NA_VACANCY_FRACTION,
        "na_removed_for_md": int(md_na_removed),
        "cif_md_start": cif_md_start,
    }

    log(
        f"[MD] Starting MD with {potential.upper()} | "
        f"T = {MD_TEMP_K} K | dt = {MD_TIMESTEP_FS} fs | steps = {MD_STEPS}"
    )

    for step in range(1, MD_STEPS + 1):
        if should_stop():
            log(f"[MD] Cancel requested at step {step}")
            avg_temp = float(np.mean(temp_series_k)) if temp_series_k else float(MD_TEMP_K)
            final_temp = float(temp_series_k[-1]) if temp_series_k else float(MD_TEMP_K)
            yield cancelled_payload(
                f"MD stopped by user at step {step}",
                step=step,
                steps=MD_STEPS,
                avg_temperature_k=avg_temp,
                final_temperature_k=final_temp,
                total_time_ps=step * MD_TIMESTEP_FS / 1000.0,
                cif_md_start=cif_md_start,
                na_removed_for_md=int(md_na_removed),
                na_vacancy_fraction=MD_NA_VACANCY_FRACTION,
            )
            return

        dyn.run(1)

        curr_scaled = atoms_md.get_scaled_positions(wrap=True)
        delta = curr_scaled - prev_scaled

        delta[:, 0] = delta[:, 0] - np.round(delta[:, 0]) if pbc[0] else delta[:, 0]
        delta[:, 1] = delta[:, 1] - np.round(delta[:, 1]) if pbc[1] else delta[:, 1]
        delta[:, 2] = delta[:, 2] - np.round(delta[:, 2]) if pbc[2] else delta[:, 2]

        cumulative_frac += delta
        prev_scaled = curr_scaled

        disp_cart = cumulative_frac @ cell
        sq = np.sum(disp_cart ** 2, axis=1)

        time_ps = step * MD_TIMESTEP_FS / 1000.0
        msd_na = float(np.mean(sq[na_mask])) if np.any(na_mask) else 0.0
        msd_non_na = float(np.mean(sq[non_na_mask])) if np.any(non_na_mask) else 0.0

        ek = atoms_md.get_kinetic_energy()
        t_inst = ek / (1.5 * units.kB * len(atoms_md))
        temp_series_k.append(float(t_inst))

        if step % MD_LOG_INTERVAL == 0 or step == 1 or step == MD_STEPS:
            percent = 100.0 * step / MD_STEPS
            log(f"[MD] step {step:5d}/{MD_STEPS} ({percent:5.1f}%)  T = {t_inst:7.1f} K")

        yield {
            "event": "progress",
            "step": step,
            "steps": MD_STEPS,
            "time_ps": float(time_ps),
            "msd_na": msd_na,
            "msd_non_na": msd_non_na,
            "temperature_k": float(t_inst),
            "progress": float(step / MD_STEPS),
        }

    avg_temp = float(np.mean(temp_series_k)) if temp_series_k else float(MD_TEMP_K)
    final_temp = float(temp_series_k[-1]) if temp_series_k else float(MD_TEMP_K)

    log("[MD] Streaming MD finished")

    yield {
        "event": "result",
        "potential": potential,
        "temperature_k": MD_TEMP_K,
        "timestep_fs": MD_TIMESTEP_FS,
        "steps": MD_STEPS,
        "sample_interval": MD_SAMPLE_INTERVAL,
        "total_time_ps": MD_STEPS * MD_TIMESTEP_FS / 1000.0,
        "n_atoms": len(atoms_md),
        "n_na_atoms": int(np.sum(na_mask)),
        "n_non_na_atoms": int(np.sum(non_na_mask)),
        "avg_temperature_k": avg_temp,
        "final_temperature_k": final_temp,
        "na_vacancy_fraction": MD_NA_VACANCY_FRACTION,
        "na_removed_for_md": int(md_na_removed),
        "cif_md_start": cif_md_start,
    }


def run_screening_stream(transition_metals, dopants, fractions, potential="uma"):
    if not transition_metals:
        raise ValueError("transition_metals list is empty")
    if not dopants:
        raise ValueError("dopants list is empty")

    potential = normalize_potential(potential)
    log(f"[RUN] Starting screening with potential = {potential.upper()}")
    log(f"[RUN] Selected transition metals: {transition_metals}")
    log(f"[RUN] Selected dopants: {dopants}")
    log(f"[RUN] Input fractions: {fractions}")

    yield {
        "event": "status",
        "message": f"Initializing {potential.upper()} calculator...",
        "progress": 0.02,
    }

    calc = get_calc(potential)

    validate_fractions(
        transition_metals=transition_metals,
        dopants=dopants,
        fractions=fractions,
    )

    yield {
        "event": "status",
        "message": "Fractions validated",
        "progress": 0.06,
    }

    log(f"[RUN] Loading template: {TEMPLATE_PATH}")
    template_atoms = read(TEMPLATE_PATH)

    tm_indices = [i for i, a in enumerate(template_atoms) if a.symbol in ["Mn", "Zr"]]
    if len(tm_indices) != 12:
        raise ValueError(f"Expected 12 TM sites (Mn/Zr), found {len(tm_indices)}")

    total_tm_sites = len(tm_indices)
    counts = fractions_to_counts(fractions, total_tm_sites)
    log(f"[RUN] TM-site counts from fractions: {counts}")

    yield {
        "event": "status",
        "message": "Generating unique configurations...",
        "progress": 0.10,
        "site_counts": counts,
    }

    signatures = generate_unique_config_signatures(counts, n_configs=N_CONFIGS)

    configuration_energies = []
    relaxed_candidates = []
    candidate_cif_doped = []

    total_relax_units = len(signatures) + 1
    relax_block_start = 0.12
    relax_block_end = 0.90
    relax_block_span = relax_block_end - relax_block_start

    log("[CONFIG] Calculating energies for generated configurations...")
    for i, signature in enumerate(signatures, start=1):
        log(f"[CONFIG] Configuration {i}/{len(signatures)}")

        atoms_candidate = apply_signature_to_atoms(template_atoms, tm_indices, signature)
        cif_doped = atoms_to_cif_string(atoms_candidate)

        yield {
            "event": "progress",
            "message": f"Relaxing configuration {i}/{len(signatures)}...",
            "stage": "configuration_relaxation",
            "config_index": i,
            "config_total": len(signatures),
            "progress": relax_block_start + relax_block_span * ((i - 1) / total_relax_units),
        }

        E_sod = relax_ase(
            atoms_candidate,
            calc=calc,
            label=f"sodiated configuration {i}",
        )

        configuration_energies.append(
            {
                "name": f"Configuration {i}",
                "index": i,
                "energy": round(float(E_sod), 6),
            }
        )
        relaxed_candidates.append((atoms_candidate, float(E_sod)))
        candidate_cif_doped.append(cif_doped)
        log(f"[CONFIG] Configuration {i} energy = {E_sod:.6f} eV")

        yield {
            "event": "config_done",
            "message": f"Finished configuration {i}/{len(signatures)}",
            "stage": "configuration_relaxation",
            "config_index": i,
            "config_total": len(signatures),
            "energy": round(float(E_sod), 6),
            "configuration_energies": configuration_energies,
            "progress": relax_block_start + relax_block_span * (i / total_relax_units),
        }

    selected_idx = min(range(len(relaxed_candidates)), key=lambda k: relaxed_candidates[k][1])
    selected_config_number = selected_idx + 1

    atoms_sod = relaxed_candidates[selected_idx][0]
    E_sod = float(relaxed_candidates[selected_idx][1])

    log(
        f"[CONFIG] Selected Configuration {selected_config_number} "
        f"with lowest sodiated energy = {E_sod:.6f} eV"
    )

    cif_doped = candidate_cif_doped[selected_idx]
    cif_sodiated_relaxed = atoms_to_cif_string(atoms_sod)

    yield {
        "event": "status",
        "message": f"Selected lowest-energy configuration: {selected_config_number}",
        "selected_configuration": {
            "name": f"Configuration {selected_config_number}",
            "index": selected_config_number,
            "energy": round(float(E_sod), 6),
        },
        "progress": 0.92,
    }

    log(f"[VOLTAGE] Removing exactly {NA_REMOVED_FIXED} Na from selected configuration...")
    atoms_des = atoms_sod.copy()
    na_indices = [i for i, a in enumerate(atoms_des) if a.symbol == "Na"]

    if len(na_indices) < NA_REMOVED_FIXED:
        raise ValueError(
            f"Structure has only {len(na_indices)} Na, cannot remove {NA_REMOVED_FIXED}"
        )

    for idx in sorted(na_indices[:NA_REMOVED_FIXED], reverse=True):
        del atoms_des[idx]

    yield {
        "event": "progress",
        "message": "Relaxing desodiated selected configuration...",
        "stage": "desodiated_relaxation",
        "progress": 0.94,
    }

    E_desod = relax_ase(
        atoms_des,
        calc=calc,
        label="selected desodiated configuration",
    )
    cif_desodiated_relaxed = atoms_to_cif_string(atoms_des)

    yield {
        "event": "status",
        "message": "Computing Na reference and voltage...",
        "progress": 0.97,
    }

    mu_na = float(get_mu_na(potential))
    V = compute_voltage(E_sod, E_desod, mu_na, n_removed=NA_REMOVED_FIXED)

    log(
        f"[VOLTAGE] E_sod = {E_sod:.6f} eV | "
        f"E_desod = {E_desod:.6f} eV | "
        f"mu_Na = {mu_na:.6f} eV | "
        f"V = {V:.3f} V"
    )

    nonzero_species = {el: float(frac) for el, frac in fractions.items() if float(frac) > 0}
    tm_species, dopant_species, chosen_tm, chosen_dopant = summarize_species(
        counts, transition_metals, dopants
    )

    log("[RUN] Screening workflow completed successfully")

    result = {
        "potential": potential,
        "n_configurations": len(configuration_energies),
        "configuration_energies": configuration_energies,
        "selected_configuration": {
            "name": f"Configuration {selected_config_number}",
            "index": selected_config_number,
            "energy": round(float(E_sod), 6),
        },
        "chosen_tm": chosen_tm,
        "chosen_dopant": chosen_dopant,
        "tm_sites": int(sum(tm_species.values())),
        "dopant_sites": int(sum(dopant_species.values())),
        "na_removed": int(NA_REMOVED_FIXED),
        "mu_na": float(mu_na),
        "sodiated_energy": float(E_sod),
        "desodiated_energy": float(E_desod),
        "voltage": round(float(V), 3),
        "composition": nonzero_species,
        "site_counts": counts,
        "cif_doped": cif_doped,
        "cif_sodiated_relaxed": cif_sodiated_relaxed,
        "cif_desodiated_relaxed": cif_desodiated_relaxed,
    }

    yield {
        "event": "result",
        "message": "Screening completed",
        "progress": 1.0,
        **result,
    }


def run_screening(transition_metals, dopants, fractions, potential="uma"):
    final_result = None
    for item in run_screening_stream(
        transition_metals=transition_metals,
        dopants=dopants,
        fractions=fractions,
        potential=potential,
    ):
        if item.get("event") == "result":
            final_result = {k: v for k, v in item.items() if k not in {"event", "message", "progress"}}

    if final_result is None:
        raise RuntimeError("Screening did not produce a final result")

    return final_result

def run_relaxation_stream(
    filename: str,
    file_bytes: bytes,
    potential="uma",
    optimizer="LBFGS",
    fmax=0.05,
    steps=300,
):
    potential = normalize_potential(potential)
    calc = get_calc(potential)

    atoms = uploaded_file_to_atoms(filename, file_bytes)
    atoms.calc = calc

    initial_cif = atoms_to_cif_string(atoms)
    initial_energy = float(atoms.get_potential_energy())

    yield {
        "event": "meta",
        "filename": filename,
        "potential": potential,
        "optimizer": optimizer,
        "n_atoms": len(atoms),
        "initial_cif": initial_cif,
        "initial_energy": initial_energy,
    }

    traj_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".traj")
    traj_tmp.close()
    traj_path = traj_tmp.name

    traj = Trajectory(traj_path, "w", atoms)
    dyn = get_optimizer(optimizer, atoms, logfile="-")

    for step in range(steps):
        dyn.run(fmax=fmax, steps=1)
        traj.write(atoms)

        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        max_force = float(np.abs(forces).max()) if len(forces) else 0.0

        yield {
            "event": "progress",
            "step": step + 1,
            "steps": steps,
            "energy": energy,
            "max_force": max_force,
            "progress": float(step + 1) / float(steps),
            "log_line": f"step {step + 1}/{steps} | E = {energy:.6f} eV | max|F| = {max_force:.4f} eV/A",
        }

        if max_force <= fmax:
            break

    traj.close()

    final_energy = float(atoms.get_potential_energy())
    relaxed_cif = atoms_to_cif_string(atoms)

    yield {
        "event": "result",
        "filename": filename,
        "potential": potential,
        "optimizer": optimizer,
        "n_atoms": len(atoms),
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "relaxed_cif": relaxed_cif,
        "traj_path": traj_path,
    }



def run_uploaded_md_stream(
    filename: str,
    file_bytes: bytes,
    potential="uma",
    temperature_k=800.0,
    timestep_fs=1.0,
    total_time_ps=5.0,
    should_stop: Callable[[], bool] | None = None,
):
    potential = normalize_potential(potential)
    should_stop = should_stop or (lambda: False)

    atoms = uploaded_file_to_atoms(filename, file_bytes)

    yield {
        "event": "status",
        "message": f"Uploaded structure '{filename}' parsed successfully",
    }

    calc = get_calc(potential)
    atoms.calc = calc

    steps = max(1, int(round(float(total_time_ps) * 1000.0 / float(timestep_fs))))

    MaxwellBoltzmannDistribution(atoms, temperature_K=float(temperature_k))

    dyn = Langevin(
        atoms,
        timestep=float(timestep_fs) * units.fs,
        temperature_K=float(temperature_k),
        friction=md_friction_for_potential(potential),
    )

    pbc = atoms.get_pbc()
    cell = np.array(atoms.get_cell())

    ref_scaled = atoms.get_scaled_positions(wrap=True)
    prev_scaled = ref_scaled.copy()
    cumulative_frac = np.zeros_like(ref_scaled)

    temp_series_k = []

    initial_cif = atoms_to_cif_string(atoms)
    species_symbols = [a.symbol for a in atoms]
    unique_species = sorted(set(species_symbols))

    traj_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".traj")
    traj_tmp.close()
    traj_path = traj_tmp.name
    traj = Trajectory(traj_path, "w", atoms)

    yield {
        "event": "meta",
        "filename": filename,
        "potential": potential,
        "temperature_k": float(temperature_k),
        "timestep_fs": float(timestep_fs),
        "total_time_ps": float(total_time_ps),
        "steps": steps,
        "n_atoms": len(atoms),
        "initial_cif": initial_cif,
        "species": unique_species,
    }

    for step in range(1, steps + 1):
        if should_stop():
            avg_temp = float(np.mean(temp_series_k)) if temp_series_k else float(temperature_k)
            traj.close()
            yield {
                "event": "cancelled",
                "message": f"Upload MD stopped by user at step {step}",
                "avg_temperature_k": avg_temp,
                "step": step,
                "steps": steps,
                "traj_path": traj_path,
            }
            return

        dyn.run(1)
        traj.write(atoms)

        curr_scaled = atoms.get_scaled_positions(wrap=True)
        delta = curr_scaled - prev_scaled

        delta[:, 0] = delta[:, 0] - np.round(delta[:, 0]) if pbc[0] else delta[:, 0]
        delta[:, 1] = delta[:, 1] - np.round(delta[:, 1]) if pbc[1] else delta[:, 1]
        delta[:, 2] = delta[:, 2] - np.round(delta[:, 2]) if pbc[2] else delta[:, 2]

        cumulative_frac += delta
        prev_scaled = curr_scaled

        disp_cart = cumulative_frac @ cell
        sq = np.sum(disp_cart ** 2, axis=1)

        time_ps = step * float(timestep_fs) / 1000.0

        msd_by_species = {}
        for sp in unique_species:
            mask = np.array([a.symbol == sp for a in atoms], dtype=bool)
            msd_by_species[sp] = float(np.mean(sq[mask])) if np.any(mask) else 0.0

        ek = atoms.get_kinetic_energy()
        t_inst = ek / (1.5 * units.kB * len(atoms))
        temp_series_k.append(float(t_inst))

        yield {
            "event": "progress",
            "step": step,
            "steps": steps,
            "time_ps": float(time_ps),
            "msd_by_species": msd_by_species,
            "temperature_k": float(t_inst),
            "progress": float(step / steps),
        }

    traj.close()

    final_cif = atoms_to_cif_string(atoms)

    yield {
        "event": "result",
        "filename": filename,
        "potential": potential,
        "temperature_k": float(temperature_k),
        "timestep_fs": float(timestep_fs),
        "total_time_ps": float(total_time_ps),
        "steps": steps,
        "avg_temperature_k": float(np.mean(temp_series_k)) if temp_series_k else float(temperature_k),
        "final_temperature_k": float(temp_series_k[-1]) if temp_series_k else float(temperature_k),
        "final_cif": final_cif,
        "traj_path": traj_path,
    }