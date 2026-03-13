from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter

import io
import random
from functools import lru_cache

import numpy as np

from ase.io import read
from ase.build import bulk
from ase.optimize import LBFGS
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
MD_SAMPLE_INTERVAL = 1          # live update every MD step
MD_NA_VACANCY_FRACTION = 0.25
MD_LOG_INTERVAL = 100

SUPPORTED_POTENTIALS = {"uma", "orb"}


def log(msg: str):
    print(msg, flush=True)


def atoms_to_cif_string(atoms) -> str:
    struct = AseAtomsAdaptor.get_structure(atoms)
    return str(CifWriter(struct))


def cif_string_to_atoms(cif_text: str):
    if not cif_text or not cif_text.strip():
        raise ValueError("CIF input is empty")
    return read(io.StringIO(cif_text), format="cif")


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
    return -(E_desod - E_sod + n_removed * mu_na) / n_removed


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


def run_md_stream(cif: str, potential="uma"):
    potential = normalize_potential(potential)
    log(f"[MD] Starting streaming MD workflow with {potential.upper()}")

    calc = get_calc(potential)
    atoms = cif_string_to_atoms(cif)

    yield {
        "event": "status",
        "message": "Loaded selected structure for MD",
        "potential": potential,
    }

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


def run_screening(transition_metals, dopants, fractions, potential="uma"):
    if not transition_metals:
        raise ValueError("transition_metals list is empty")
    if not dopants:
        raise ValueError("dopants list is empty")

    potential = normalize_potential(potential)
    log(f"[RUN] Starting screening with potential = {potential.upper()}")
    log(f"[RUN] Selected transition metals: {transition_metals}")
    log(f"[RUN] Selected dopants: {dopants}")
    log(f"[RUN] Input fractions: {fractions}")

    calc = get_calc(potential)

    validate_fractions(
        transition_metals=transition_metals,
        dopants=dopants,
        fractions=fractions,
    )

    log("[RUN] Fractions validated")

    log(f"[RUN] Loading template: {TEMPLATE_PATH}")
    template_atoms = read(TEMPLATE_PATH)

    tm_indices = [i for i, a in enumerate(template_atoms) if a.symbol in ["Mn", "Zr"]]
    if len(tm_indices) != 12:
        raise ValueError(f"Expected 12 TM sites (Mn/Zr), found {len(tm_indices)}")

    total_tm_sites = len(tm_indices)
    counts = fractions_to_counts(fractions, total_tm_sites)
    log(f"[RUN] TM-site counts from fractions: {counts}")

    signatures = generate_unique_config_signatures(counts, n_configs=N_CONFIGS)

    configuration_energies = []
    relaxed_candidates = []
    candidate_cif_doped = []

    log("[CONFIG] Calculating energies for generated configurations...")
    for i, signature in enumerate(signatures, start=1):
        log(f"[CONFIG] Configuration {i}/{len(signatures)}")
        atoms_candidate = apply_signature_to_atoms(template_atoms, tm_indices, signature)
        cif_doped = atoms_to_cif_string(atoms_candidate)

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

    log(f"[VOLTAGE] Removing exactly {NA_REMOVED_FIXED} Na from selected configuration...")
    atoms_des = atoms_sod.copy()
    na_indices = [i for i, a in enumerate(atoms_des) if a.symbol == "Na"]

    if len(na_indices) < NA_REMOVED_FIXED:
        raise ValueError(
            f"Structure has only {len(na_indices)} Na, cannot remove {NA_REMOVED_FIXED}"
        )

    for idx in sorted(na_indices[:NA_REMOVED_FIXED], reverse=True):
        del atoms_des[idx]

    E_desod = relax_ase(
        atoms_des,
        calc=calc,
        label="selected desodiated configuration",
    )
    cif_desodiated_relaxed = atoms_to_cif_string(atoms_des)

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

    return {
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