# DrHouse

A multi-agent clinical communications assistant for a clinic. It
triages incoming patient messages (email or phone), grounds responses in the
patient's EMR history, and escalates anything outside its scope to a physician.

This is a prototype: EMR data, patient records, and tool integrations (email,
appointments, lab results) are simulated.

## Architecture

```
Patient Message (email/phone)
         |
    [IntakeAgent]          -- classifies intent and urgency
         |
    [MemoryAgent]          -- retrieves relevant EMR context via RAG
         |
    [ReasoningAgent]       -- synthesizes context + message -> draft response
         |
    [GuardrailLayer]       -- privacy, safety, and compliance checks
         |
    [PlanningEngine]       -- routes based on guardrail results + urgency
    /         |         \
[Response] [Escalation] [SafetyBlock]
```

| Component | Implementation |
|---|---|
| Memory | ChromaDB vector store + RAG retrieval over EMR notes |
| Planning | Conditional routing in LangGraph |
| Reasoning | Context-grounded chain-of-thought over retrieved EMR data |
| Tools | EMR lookup, lab results, appointments, email, voicemail transcription |
| Guardrails | PII, clinical-safety, and scope checks; runs after generation, before sending |
| Human-in-the-loop | Escalation checkpoint for urgent or out-of-scope cases |
| Short-term memory | LangGraph `MemorySaver` (per-conversation state) |
| Long-term memory | ChromaDB (patient EMR history) |

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
python main.py       # CLI demo (runs sample scenarios)
```

## Running the API

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs are served at `/docs` (Swagger) and `/redoc`.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness/readiness check |
| `/api/messages` | POST | Run a patient message through the full agent pipeline |
| `/api/patients/{patient_id}/summary` | GET | RAG-retrieved EMR summary for a patient |

Example request:

```bash
curl -X POST http://localhost:8000/api/messages \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "P001", "channel": "email", "message": "Should I still be taking the Budesonide?"}'
```

### Docker

```bash
docker build -t drhouse .
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key_here drhouse
```

### Deploying (Render, Fly.io, Railway, etc.)

The included `Procfile` and `Dockerfile` both target `api:app` and read
configuration from environment variables (see `.env.example`):

- `OPENAI_API_KEY` (required)
- `CHROMA_PERSIST_DIR` (optional; defaults to a fresh temp directory per process)
- `CORS_ORIGINS` (optional; comma-separated, defaults to `*`)
- `PORT` (used by the `Procfile` entrypoint)

The service builds its EMR vector store from synthetic data at startup, so no
external database is required to deploy.

## Project Structure

```
drhouse/
├── main.py                          # CLI entry point (demo scenarios)
├── api.py                           # FastAPI HTTP interface
├── Dockerfile
├── Procfile
├── requirements.txt
├── memory/
│   └── emr_memory.py               # Long-term EMR memory (RAG)
├── agents/
│   └── clinical_agents.py          # All agents + LangGraph graph
├── guardrails/
│   └── clinical_guardrails.py      # Safety and compliance layer
├── tools/                          # Tool-calling pipelines (extensible)
└── tests/                          # Test scenarios
```

## Design Notes

**RAG for memory.** A patient's full history won't fit in a context window, so
the memory agent retrieves only the chunks relevant to the current message
rather than loading everything.

**Guardrails run separately from reasoning.** They check the response after
it's generated and before it's sent, rather than being folded into the
reasoning agent's own logic. Keeping them separate makes it possible to update
safety rules without touching the reasoning prompt, and keeps a guardrail
trigger from corrupting quality signals if this is ever evaluated offline.

**Human-in-the-loop for anything urgent or out of scope.** The escalation
agent acknowledges the patient and hands off to a physician rather than
attempting to answer clinical questions autonomously.

**Typed state between agents.** `TypedDict` state enforces a contract between
agents so nothing can write unexpected fields, which keeps the pipeline
debuggable as it grows.
