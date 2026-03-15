from typing import Dict, List
from uuid import uuid4
import json

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, PlainTextResponse
from pydantic import BaseModel

from model import (
    run_screening,
    run_md_stream,
    run_screening_stream,
    generate_solid_state_synthesis_route,
    run_relaxation_stream,
    run_uploaded_md_stream,
    uploaded_file_to_atoms,
    atoms_to_cif_string,
)

app = FastAPI(title="Na Layered Cathode Screening API")

RELAX_UPLOAD_SESSIONS: Dict[str, Dict] = {}
UPLOAD_MD_SESSIONS: Dict[str, Dict] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:3001", "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MD_SESSIONS: Dict[str, Dict] = {}
SCREENING_SESSIONS: Dict[str, Dict] = {}
RELAX_UPLOAD_SESSIONS: Dict[str, Dict] = {}
UPLOAD_MD_SESSIONS: Dict[str, Dict] = {}
RELAX_RESULTS: Dict[str, Dict] = {}
UPLOAD_MD_RESULTS: Dict[str, Dict] = {}


class ScreeningRequest(BaseModel):
    transition_metals: List[str]
    dopants: List[str]
    fractions: Dict[str, float]
    potential: str = "uma"


class MDRequest(BaseModel):
    cif: str
    potential: str = "uma"


class SynthesisRouteRequest(BaseModel):
    composition: Dict[str, float]
    batch_mmol: float = 10.0
    na_excess_fraction: float = 0.05


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
def run(req: ScreeningRequest):
    return run_screening(
        transition_metals=req.transition_metals,
        dopants=req.dopants,
        fractions=req.fractions,
        potential=req.potential,
    )

@app.post("/preview-structure")
async def preview_structure(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        atoms = uploaded_file_to_atoms(file.filename or "structure.cif", content)
        cif_text = atoms_to_cif_string(atoms)
        return {
            "filename": file.filename or "structure.cif",
            "n_atoms": len(atoms),
            "cif": cif_text,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse structure: {e}")

@app.post("/run-session")
def create_screening_session(req: ScreeningRequest):
    session_id = uuid4().hex
    SCREENING_SESSIONS[session_id] = {
        "transition_metals": req.transition_metals,
        "dopants": req.dopants,
        "fractions": req.fractions,
        "potential": req.potential,
    }
    return {"session_id": session_id}


@app.get("/run-stream/{session_id}")
def run_screening_stream_route(session_id: str):
    payload = SCREENING_SESSIONS.pop(session_id, None)
    if payload is None:
        raise HTTPException(status_code=404, detail="Screening session not found or already used")

    def event_stream():
        try:
            yield "event: status\n"
            yield "data: " + json.dumps({"message": "Screening stream started"}) + "\n\n"

            for item in run_screening_stream(
                transition_metals=payload["transition_metals"],
                dopants=payload["dopants"],
                fractions=payload["fractions"],
                potential=payload["potential"],
            ):
                event = item.get("event", "progress")
                yield f"event: {event}\n"
                yield "data: " + json.dumps(item) + "\n\n"

            yield "event: done\n"
            yield "data: " + json.dumps({"message": "Screening stream completed"}) + "\n\n"

        except Exception as e:
            yield "event: error\n"
            yield "data: " + json.dumps({"error": str(e)}) + "\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/run-md-session")
def create_md_session(req: MDRequest):
    if not req.cif.strip():
        raise HTTPException(status_code=400, detail="CIF input is empty")

    session_id = uuid4().hex
    MD_SESSIONS[session_id] = {
        "cif": req.cif,
        "potential": req.potential,
        "cancelled": False,
        "consumed": False,
    }
    return {"session_id": session_id}


@app.post("/stop-md/{session_id}")
def stop_md(session_id: str):
    payload = MD_SESSIONS.get(session_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="MD session not found")
    payload["cancelled"] = True
    return {"status": "stopping", "session_id": session_id}


@app.get("/run-md-stream/{session_id}")
def run_md_stream_route(session_id: str):
    payload = MD_SESSIONS.get(session_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="MD session not found")

    if payload.get("consumed"):
        raise HTTPException(status_code=404, detail="MD session already used")

    payload["consumed"] = True

    def should_stop():
        session = MD_SESSIONS.get(session_id)
        if session is None:
            return True
        return bool(session.get("cancelled", False))

    def event_stream():
        try:
            yield "event: status\n"
            yield "data: " + json.dumps({"message": "MD stream started"}) + "\n\n"

            for item in run_md_stream(
                cif=payload["cif"],
                potential=payload["potential"],
                should_stop=should_stop,
            ):
                event = item.get("event", "progress")
                yield f"event: {event}\n"
                yield "data: " + json.dumps(item) + "\n\n"

            yield "event: done\n"
            yield "data: " + json.dumps({"message": "MD stream completed"}) + "\n\n"

        except Exception as e:
            yield "event: error\n"
            yield "data: " + json.dumps({"error": str(e)}) + "\n\n"
        finally:
            MD_SESSIONS.pop(session_id, None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/generate-synthesis-route")
def generate_synthesis_route(req: SynthesisRouteRequest):
    return generate_solid_state_synthesis_route(
        composition=req.composition,
        batch_mmol=req.batch_mmol,
        na_excess_fraction=req.na_excess_fraction,
    )

@app.post("/relax-upload-session")
async def create_relax_upload_session(
    file: UploadFile = File(...),
    potential: str = Form("uma"),
    optimizer: str = Form("LBFGS"),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    session_id = uuid4().hex
    RELAX_UPLOAD_SESSIONS[session_id] = {
        "filename": file.filename or "structure.cif",
        "content": content,
        "potential": potential,
        "optimizer": optimizer,
    }
    return {"session_id": session_id}


@app.get("/relax-upload-stream/{session_id}")
def relax_upload_stream(session_id: str):
    payload = RELAX_UPLOAD_SESSIONS.pop(session_id, None)
    if payload is None:
        raise HTTPException(status_code=404, detail="Relax upload session not found or already used")

    def event_stream():
        try:
            result_id = uuid4().hex

            for item in run_relaxation_stream(
                filename=payload["filename"],
                file_bytes=payload["content"],
                potential=payload["potential"],
                optimizer=payload["optimizer"],
            ):
                if item.get("event") == "result":
                    RELAX_RESULTS[result_id] = {
                        "relaxed_cif": item["relaxed_cif"],
                        "traj_path": item["traj_path"],
                    }
                    item["result_id"] = result_id

                event = item.get("event", "progress")
                yield f"event: {event}\n"
                yield "data: " + json.dumps(item) + "\n\n"

            yield "event: done\n"
            yield "data: " + json.dumps({"message": "Relaxation completed"}) + "\n\n"

        except Exception as e:
            yield "event: error\n"
            yield "data: " + json.dumps({"error": str(e)}) + "\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/download-relaxed-cif/{result_id}")
def download_relaxed_cif(result_id: str):
    payload = RELAX_RESULTS.get(result_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Relaxation result not found")

    return PlainTextResponse(
        payload["relaxed_cif"],
        media_type="text/plain",
        headers={"Content-Disposition": 'attachment; filename="relaxed_structure.cif"'},
    )


@app.get("/download-relax-traj/{result_id}")
def download_relax_traj(result_id: str):
    payload = RELAX_RESULTS.get(result_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Relaxation result not found")

    return FileResponse(
        payload["traj_path"],
        filename="relaxation.traj",
        media_type="application/octet-stream",
    )


@app.post("/relax-upload-session")
async def create_relax_upload_session(
    file: UploadFile = File(...),
    potential: str = Form("uma"),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    session_id = uuid4().hex
    RELAX_UPLOAD_SESSIONS[session_id] = {
        "filename": file.filename or "structure.cif",
        "content": content,
        "potential": potential,
    }
    return {"session_id": session_id}


@app.get("/relax-upload-stream/{session_id}")
def relax_upload_stream(session_id: str):
    payload = RELAX_UPLOAD_SESSIONS.pop(session_id, None)
    if payload is None:
        raise HTTPException(status_code=404, detail="Relax upload session not found or already used")

    def event_stream():
        try:
            for item in run_relaxation_stream(
                filename=payload["filename"],
                file_bytes=payload["content"],
                potential=payload["potential"],
            ):
                event = item.get("event", "progress")
                yield f"event: {event}\n"
                yield "data: " + json.dumps(item) + "\n\n"
            yield "event: done\n"
            yield "data: " + json.dumps({"message": "Relaxation completed"}) + "\n\n"
        except Exception as e:
            yield "event: error\n"
            yield "data: " + json.dumps({"error": str(e)}) + "\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/md-upload-session")
async def create_md_upload_session(
    file: UploadFile = File(...),
    potential: str = Form("uma"),
    temperature_k: float = Form(800.0),
    timestep_fs: float = Form(1.0),
    total_time_ps: float = Form(5.0),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    session_id = uuid4().hex
    UPLOAD_MD_SESSIONS[session_id] = {
        "filename": file.filename or "structure.cif",
        "content": content,
        "potential": potential,
        "temperature_k": temperature_k,
        "timestep_fs": timestep_fs,
        "total_time_ps": total_time_ps,
        "cancelled": False,
        "consumed": False,
    }
    return {"session_id": session_id}


@app.post("/stop-upload-md/{session_id}")
def stop_upload_md(session_id: str):
    payload = UPLOAD_MD_SESSIONS.get(session_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Upload MD session not found")
    payload["cancelled"] = True
    return {"status": "stopping", "session_id": session_id}


@app.get("/md-upload-stream/{session_id}")
def md_upload_stream(session_id: str):
    payload = UPLOAD_MD_SESSIONS.get(session_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Upload MD session not found")

    if payload.get("consumed"):
        raise HTTPException(status_code=404, detail="Upload MD session already used")

    payload["consumed"] = True

    def should_stop():
        session = UPLOAD_MD_SESSIONS.get(session_id)
        if session is None:
          return True
        return bool(session.get("cancelled", False))

    def event_stream():
        try:
            for item in run_uploaded_md_stream(
                filename=payload["filename"],
                file_bytes=payload["content"],
                potential=payload["potential"],
                temperature_k=payload["temperature_k"],
                timestep_fs=payload["timestep_fs"],
                total_time_ps=payload["total_time_ps"],
                should_stop=should_stop,
            ):
                event = item.get("event", "progress")
                yield f"event: {event}\n"
                yield "data: " + json.dumps(item) + "\n\n"

            yield "event: done\n"
            yield "data: " + json.dumps({"message": "Upload MD completed"}) + "\n\n"
        except Exception as e:
            yield "event: error\n"
            yield "data: " + json.dumps({"error": str(e)}) + "\n\n"
        finally:
            UPLOAD_MD_SESSIONS.pop(session_id, None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.post("/md-upload-session")
async def create_md_upload_session(
    file: UploadFile = File(...),
    potential: str = Form("uma"),
    temperature_k: float = Form(800.0),
    timestep_fs: float = Form(1.0),
    total_time_ps: float = Form(5.0),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    session_id = uuid4().hex
    UPLOAD_MD_SESSIONS[session_id] = {
        "filename": file.filename or "structure.cif",
        "content": content,
        "potential": potential,
        "temperature_k": temperature_k,
        "timestep_fs": timestep_fs,
        "total_time_ps": total_time_ps,
        "cancelled": False,
        "consumed": False,
    }
    return {"session_id": session_id}


@app.post("/stop-upload-md/{session_id}")
def stop_upload_md(session_id: str):
    payload = UPLOAD_MD_SESSIONS.get(session_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Upload MD session not found")
    payload["cancelled"] = True
    return {"status": "stopping", "session_id": session_id}


@app.get("/md-upload-stream/{session_id}")
def md_upload_stream(session_id: str):
    payload = UPLOAD_MD_SESSIONS.get(session_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Upload MD session not found")

    if payload.get("consumed"):
        raise HTTPException(status_code=404, detail="Upload MD session already used")

    payload["consumed"] = True

    def should_stop():
        session = UPLOAD_MD_SESSIONS.get(session_id)
        if session is None:
            return True
        return bool(session.get("cancelled", False))

    def event_stream():
        try:
            result_id = uuid4().hex

            for item in run_uploaded_md_stream(
                filename=payload["filename"],
                file_bytes=payload["content"],
                potential=payload["potential"],
                temperature_k=payload["temperature_k"],
                timestep_fs=payload["timestep_fs"],
                total_time_ps=payload["total_time_ps"],
                should_stop=should_stop,
            ):
                if item.get("event") == "result":
                    UPLOAD_MD_RESULTS[result_id] = {
                        "final_cif": item["final_cif"],
                        "traj_path": item["traj_path"],
                    }
                    item["result_id"] = result_id

                event = item.get("event", "progress")
                yield f"event: {event}\n"
                yield "data: " + json.dumps(item) + "\n\n"

            yield "event: done\n"
            yield "data: " + json.dumps({"message": "Upload MD completed"}) + "\n\n"

        except Exception as e:
            yield "event: error\n"
            yield "data: " + json.dumps({"error": str(e)}) + "\n\n"
        finally:
            UPLOAD_MD_SESSIONS.pop(session_id, None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/download-upload-md-traj/{result_id}")
def download_upload_md_traj(result_id: str):
    payload = UPLOAD_MD_RESULTS.get(result_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Upload MD result not found")

    return FileResponse(
        payload["traj_path"],
        filename="uploaded_md.traj",
        media_type="application/octet-stream",
    )


@app.get("/download-upload-md-cif/{result_id}")
def download_upload_md_cif(result_id: str):
    payload = UPLOAD_MD_RESULTS.get(result_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Upload MD result not found")

    return PlainTextResponse(
        payload["final_cif"],
        media_type="text/plain",
        headers={"Content-Disposition": 'attachment; filename="uploaded_md_final.cif"'},
    )