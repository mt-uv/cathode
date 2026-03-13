from typing import Dict, List
from uuid import uuid4
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from model import run_screening, run_md_stream

app = FastAPI(title="Na Layered Cathode Screening API")

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

MD_SESSIONS: Dict[str, Dict[str, str]] = {}


class ScreeningRequest(BaseModel):
    transition_metals: List[str]
    dopants: List[str]
    fractions: Dict[str, float]
    potential: str = "uma"


class MDRequest(BaseModel):
    cif: str
    potential: str = "uma"


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


@app.post("/run-md-session")
def create_md_session(req: MDRequest):
    if not req.cif.strip():
        raise HTTPException(status_code=400, detail="CIF input is empty")

    session_id = uuid4().hex
    MD_SESSIONS[session_id] = {
        "cif": req.cif,
        "potential": req.potential,
    }
    return {"session_id": session_id}


@app.get("/run-md-stream/{session_id}")
def run_md_stream_route(session_id: str):
    payload = MD_SESSIONS.pop(session_id, None)
    if payload is None:
        raise HTTPException(status_code=404, detail="MD session not found or already used")

    def event_stream():
        try:
            yield "event: status\ndata: " + json.dumps({
                "message": "MD stream started"
            }) + "\n\n"

            for item in run_md_stream(
                cif=payload["cif"],
                potential=payload["potential"],
            ):
                event = item.get("event", "progress")
                yield f"event: {event}\n"
                yield "data: " + json.dumps(item) + "\n\n"

            yield "event: done\n"
            yield "data: " + json.dumps({"message": "MD stream completed"}) + "\n\n"

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