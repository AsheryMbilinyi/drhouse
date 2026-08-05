"""
api.py -- FastAPI HTTP interface for DrHouse

Exposes the multi-agent clinical communications pipeline as a REST API,
suitable for containerized deployment behind a load balancer.

Usage:
    export OPENAI_API_KEY=your_key_here
    uvicorn api:app --host 0.0.0.0 --port 8000

Production:
    gunicorn api:app -k uvicorn.workers.UvicornWorker -w 2 --bind 0.0.0.0:$PORT
"""

import json
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("drhouse")

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY environment variable is required to start DrHouse."
    )

from memory.emr_memory import EMRMemorySystem, SYNTHETIC_EMR_NOTES
from agents.clinical_agents import build_agent_graph
from tools.clinical_tools import transcribe_voice_message

# Each process gets its own persistent directory by default so repeated
# startups don't accumulate duplicate embeddings in a shared volume.
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR") or tempfile.mkdtemp(prefix="drhouse_chroma_")

app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Building EMR memory system...")
    memory = EMRMemorySystem(persist_directory=CHROMA_DIR)
    memory.build_memory(SYNTHETIC_EMR_NOTES)

    logger.info("Assembling multi-agent graph...")
    app_state["memory"] = memory
    app_state["graph"] = build_agent_graph(memory)
    logger.info("DrHouse ready.")

    yield

    app_state.clear()


app = FastAPI(
    title="DrHouse API",
    description="Multi-agent clinical communications system for Pacific Digestive Health.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


class MessageRequest(BaseModel):
    patient_id: str = Field(..., examples=["P001"])
    channel: Literal["email", "phone"] = "email"
    message: str = Field(
        "", description="Patient message text. Ignored for phone channel if audio_file is set."
    )
    audio_file: Optional[str] = Field(
        None, description="Voicemail filename to transcribe, for channel='phone'."
    )
    thread_id: Optional[str] = Field(
        None, description="Conversation thread id for multi-turn context. Generated if omitted."
    )


class MessageResponse(BaseModel):
    thread_id: str
    intent: str
    urgency: str
    guardrail_passed: bool
    risk_level: str
    requires_physician_review: bool
    final_response: str
    escalation_note: str


class PatientSummaryResponse(BaseModel):
    patient_id: str
    summary: str


@app.get("/health")
def health():
    return {"status": "ok", "ready": "graph" in app_state}


@app.post("/api/messages", response_model=MessageResponse)
def process_message(req: MessageRequest):
    if "graph" not in app_state:
        raise HTTPException(status_code=503, detail="Service not ready")

    message = req.message
    if req.channel == "phone":
        transcription = transcribe_voice_message.invoke({
            "audio_file_path": req.audio_file or "voicemail_001.mp3",
            "language": "en",
        })
        message = json.loads(transcription).get("text", req.message)

    thread_id = req.thread_id or str(uuid.uuid4())

    initial_state = {
        "patient_message": message,
        "patient_id": req.patient_id,
        "communication_channel": req.channel,
        "intent": "", "urgency": "", "emr_context": "",
        "draft_response": "", "guardrail_passed": False,
        "guardrail_reason": "", "requires_physician_review": False,
        "risk_level": "low", "final_response": "", "escalation_note": "",
    }

    try:
        result = app_state["graph"].invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception:
        logger.exception("Pipeline failed for patient %s", req.patient_id)
        raise HTTPException(status_code=500, detail="Failed to process message")

    return MessageResponse(
        thread_id=thread_id,
        intent=result.get("intent", ""),
        urgency=result.get("urgency", ""),
        guardrail_passed=result.get("guardrail_passed", False),
        risk_level=result.get("risk_level", "low"),
        requires_physician_review=result.get("requires_physician_review", False),
        final_response=result.get("final_response", ""),
        escalation_note=result.get("escalation_note", ""),
    )


@app.get("/api/patients/{patient_id}/summary", response_model=PatientSummaryResponse)
def patient_summary(patient_id: str):
    if "memory" not in app_state:
        raise HTTPException(status_code=503, detail="Service not ready")

    summary = app_state["memory"].get_patient_summary(patient_id)
    return PatientSummaryResponse(patient_id=patient_id, summary=summary)
