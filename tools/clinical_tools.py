"""
tools/clinical_tools.py

TOOL-CALLING PIPELINES -- External system integrations.

Key concepts:
- Tools are functions the agent can CALL to interact with external systems
- Each tool has a clear input schema and output schema
- Tools are registered with the LLM so it can decide WHEN to call them
- The agent uses ReAct pattern: Reason -> Act (call tool) -> Observe (result) -> Reason again
- In production: these connect to real APIs (Twilio, SMTP, EMR system)
- In this demo: we simulate the external calls with realistic outputs

Tool-calling pipeline:
  Agent decides it needs information
       |
  Selects appropriate tool
       |
  Calls tool with structured arguments
       |
  Observes the result
       |
  Continues reasoning with new information

Tools implemented:
1. EmailSender         -- sends emails to patients
2. PhoneTranscriber    -- transcribes incoming voice messages (Whisper simulation)
3. EMRLookupTool       -- structured EMR data queries
4. AppointmentTool     -- checks and books appointments
5. LabResultsTool      -- retrieves lab/test results
"""

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timedelta
import json
import random


# ── TOOL INPUT SCHEMAS ────────────────────────────────────────────────────────
# Design rationale:
# Pydantic schemas define what arguments each tool accepts.
# The LLM sees these schemas and knows exactly how to call each tool.
# Strong typing prevents the agent from hallucinating arguments.

class EmailInput(BaseModel):
    to_patient_id: str = Field(description="Patient ID to send email to")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content")
    cc_physician: bool = Field(default=False, description="Whether to CC the physician")


class PhoneTranscribeInput(BaseModel):
    audio_file_path: str = Field(description="Path to audio file to transcribe")
    language: str = Field(default="en", description="Language code")


class EMRLookupInput(BaseModel):
    patient_id: str = Field(description="Patient ID to look up")
    query_type: str = Field(description="Type of query: 'medications', 'diagnoses', 'allergies', 'recent_visits', 'upcoming_appointments'")


class AppointmentInput(BaseModel):
    patient_id: str = Field(description="Patient ID")
    action: str = Field(description="Action: 'check_next' or 'book_urgent'")
    reason: Optional[str] = Field(default=None, description="Reason for appointment if booking")


class LabResultsInput(BaseModel):
    patient_id: str = Field(description="Patient ID")
    test_name: Optional[str] = Field(default=None, description="Specific test name, or None for all recent results")


# ── TOOL 1: EMAIL SENDER ─────────────────────────────────────────────────────
@tool("send_patient_email", args_schema=EmailInput)
def send_patient_email(
    to_patient_id: str,
    subject: str,
    body: str,
    cc_physician: bool = False
) -> str:
    """
    Send an email to a patient.
    Use this tool when you need to communicate with a patient via email.

    Design rationale:
    In production this connects to SMTP or an email API (SendGrid, Mailgun).
    The tool abstracts the communication channel -- the agent doesn't need
    to know HOW email works, just WHAT to send.

    PHIPA consideration: all emails are logged with timestamp, recipient,
    and sender for audit trail compliance.
    """
    # Simulate email sending (in production: real SMTP call)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Simulate patient email addresses
    patient_emails = {
        "P001": "john.smith@email.com",
        "P002": "sarah.chen@email.com",
        "P003": "robert.martinez@email.com",
    }

    recipient_email = patient_emails.get(to_patient_id, f"{to_patient_id}@patient.com")
    cc = "dr.house@pacificdigestive.ca" if cc_physician else None

    # Audit log (in production: write to audit database)
    audit_log = {
        "action": "email_sent",
        "timestamp": timestamp,
        "to": recipient_email,
        "to_patient_id": to_patient_id,
        "cc": cc,
        "subject": subject,
        "body_length": len(body),
        "sent_by": "DrHouse"
    }

    print(f"\n  [EmailTool] Simulating email send:")
    print(f"  To: {recipient_email}")
    print(f"  Subject: {subject}")
    if cc:
        print(f"  CC: {cc}")
    print(f"  Audit log: {json.dumps(audit_log, indent=2)}")

    return json.dumps({
        "status": "sent",
        "message_id": f"MSG-{random.randint(10000, 99999)}",
        "recipient": recipient_email,
        "timestamp": timestamp,
        "audit_logged": True
    })


# ── TOOL 2: PHONE TRANSCRIBER ────────────────────────────────────────────────
@tool("transcribe_voice_message", args_schema=PhoneTranscribeInput)
def transcribe_voice_message(
    audio_file_path: str,
    language: str = "en"
) -> str:
    """
    Transcribe a patient voice message to text using Whisper.
    Use this tool when a patient leaves a voicemail or calls in.

    Design rationale:
    In production this calls OpenAI Whisper API or a local Whisper model.
    The transcription becomes the patient_message that flows through
    the rest of the agent pipeline -- enabling voice as an input channel.
    This is the multimodal (audio -> text) component of the system.
    """
    # Simulate Whisper transcription (in production: real Whisper API call)
    # We return a realistic simulated transcription

    simulated_transcriptions = {
        "voicemail_001.mp3": {
            "text": "Hi this is John Smith, patient ID P001. Um, I wanted to call because I've been having some really bad stomach cramps since yesterday and I'm also seeing some blood when I go to the bathroom. I'm a bit worried. Can someone please call me back? My number is 604-555-0123. Thanks.",
            "duration_seconds": 18.4,
            "confidence": 0.94,
            "detected_language": "en"
        },
        "voicemail_002.mp3": {
            "text": "Hello, this is Sarah Chen calling. I just wanted to ask about the low FODMAP diet that Dr. House recommended. I'm finding it really hard to know what I can and can't eat. Is there a dietitian you can refer me to? Thank you.",
            "duration_seconds": 12.1,
            "confidence": 0.97,
            "detected_language": "en"
        }
    }

    # Use simulated transcription or generate a generic one
    file_name = audio_file_path.split("/")[-1]
    if file_name in simulated_transcriptions:
        result = simulated_transcriptions[file_name]
    else:
        result = {
            "text": f"[Simulated transcription of {audio_file_path}] Patient left a message about their appointment.",
            "duration_seconds": 10.0,
            "confidence": 0.90,
            "detected_language": language
        }

    print(f"\n  [PhoneTool] Transcribed {audio_file_path}:")
    print(f"  Duration: {result['duration_seconds']}s")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Text: {result['text'][:100]}...")

    return json.dumps(result)


# ── TOOL 3: EMR LOOKUP ───────────────────────────────────────────────────────
@tool("lookup_emr_data", args_schema=EMRLookupInput)
def lookup_emr_data(
    patient_id: str,
    query_type: str
) -> str:
    """
    Look up structured data from the EMR system.
    Use this for specific structured queries: medications, allergies, diagnoses.
    For full-text semantic search over notes, use the memory system instead.

    Design rationale:
    This is a STRUCTURED tool call -- it queries specific fields in the EMR.
    Contrast with the RAG memory system which does UNSTRUCTURED semantic search.
    Both are needed: structured queries for precise data (current medications),
    semantic search for contextual understanding (what happened at last visit).
    """
    # Simulated structured EMR data
    emr_data = {
        "P001": {
            "medications": [
                {"name": "Adalimumab", "dose": "40mg", "frequency": "biweekly", "start_date": "2022-03-01"},
                {"name": "Budesonide", "dose": "9mg", "frequency": "daily", "start_date": "2024-03-15", "end_date": "2024-05-15"},
                {"name": "Vitamin D", "dose": "1000IU", "frequency": "daily", "start_date": "2023-01-01"},
            ],
            "diagnoses": [
                {"condition": "Crohn's Disease", "icd10": "K50.0", "onset": "2009-06-01", "status": "active"},
            ],
            "allergies": [
                {"allergen": "Penicillin", "reaction": "Hives", "severity": "moderate"},
            ],
            "recent_visits": [
                {"date": "2024-03-15", "type": "follow-up", "physician": "Dr. House"},
                {"date": "2023-09-10", "type": "colonoscopy", "physician": "Dr. House"},
            ],
            "upcoming_appointments": [
                {"date": "2024-09-15", "type": "colonoscopy", "physician": "Dr. House", "location": "Pacific Digestive Health"},
            ],
        },
        "P002": {
            "medications": [
                {"name": "Peppermint Oil", "dose": "0.2mL", "frequency": "TID", "start_date": "2024-01-10"},
            ],
            "diagnoses": [
                {"condition": "Irritable Bowel Syndrome (IBS), Mixed", "icd10": "K58.9", "onset": "2021-01-01", "status": "active"},
            ],
            "allergies": [],
            "recent_visits": [
                {"date": "2024-01-10", "type": "new patient", "physician": "Dr. House"},
            ],
            "upcoming_appointments": [
                {"date": "2024-02-21", "type": "follow-up", "physician": "Dr. House"},
            ],
        },
        "P003": {
            "medications": [
                {"name": "Aspirin", "dose": "81mg", "frequency": "daily", "start_date": "2023-12-01"},
            ],
            "diagnoses": [
                {"condition": "History of colorectal adenomas", "icd10": "K63.5", "status": "surveillance"},
            ],
            "allergies": [],
            "recent_visits": [
                {"date": "2023-11-05", "type": "colonoscopy + follow-up", "physician": "Dr. House"},
            ],
            "upcoming_appointments": [
                {"date": "2026-11-05", "type": "surveillance colonoscopy", "physician": "Dr. House"},
            ],
        }
    }

    patient = emr_data.get(patient_id)
    if not patient:
        return json.dumps({"error": f"No EMR data found for patient {patient_id}"})

    data = patient.get(query_type)
    if data is None:
        return json.dumps({"error": f"Query type '{query_type}' not found"})

    print(f"\n  [EMRTool] Lookup: patient={patient_id}, query={query_type}")
    print(f"  Result: {json.dumps(data, indent=2)[:200]}...")

    return json.dumps({
        "patient_id": patient_id,
        "query_type": query_type,
        "data": data,
        "retrieved_at": datetime.now().isoformat()
    })


# ── TOOL 4: APPOINTMENT TOOL ─────────────────────────────────────────────────
@tool("manage_appointment", args_schema=AppointmentInput)
def manage_appointment(
    patient_id: str,
    action: str,
    reason: Optional[str] = None
) -> str:
    """
    Check or book patient appointments.
    Use 'check_next' to find the next scheduled appointment.
    Use 'book_urgent' to schedule an urgent appointment.

    Design rationale:
    Tool calling enables the agent to TAKE ACTIONS, not just generate text.
    This is what distinguishes an agent from a chatbot -- it can modify
    state in external systems. Booking an appointment is a real-world action
    with real-world consequences, which is why it requires guardrails.
    """
    if action == "check_next":
        next_appointments = {
            "P001": {"date": "2024-09-15", "type": "colonoscopy", "location": "Pacific Digestive Health, Floor 3"},
            "P002": {"date": "2024-02-21", "type": "follow-up", "location": "Pacific Digestive Health, Suite 201"},
            "P003": {"date": "2026-11-05", "type": "surveillance colonoscopy", "location": "Pacific Digestive Health, Floor 3"},
        }
        appt = next_appointments.get(patient_id)
        if not appt:
            return json.dumps({"status": "no_upcoming", "patient_id": patient_id})

        print(f"\n  [AppointmentTool] Next appointment for {patient_id}: {appt['date']} - {appt['type']}")
        return json.dumps({"status": "found", "appointment": appt})

    elif action == "book_urgent":
        # Simulate booking an urgent appointment
        urgent_date = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        urgent_time = "10:30 AM"
        booking_ref = f"URG-{random.randint(1000, 9999)}"

        print(f"\n  [AppointmentTool] Booked urgent appointment for {patient_id}")
        print(f"  Date: {urgent_date} at {urgent_time}")
        print(f"  Reference: {booking_ref}")

        return json.dumps({
            "status": "booked",
            "patient_id": patient_id,
            "date": urgent_date,
            "time": urgent_time,
            "location": "Pacific Digestive Health, Suite 201",
            "booking_reference": booking_ref,
            "reason": reason,
            "physician": "Dr. House"
        })

    return json.dumps({"error": f"Unknown action: {action}"})


# ── TOOL 5: LAB RESULTS TOOL ─────────────────────────────────────────────────
@tool("get_lab_results", args_schema=LabResultsInput)
def get_lab_results(
    patient_id: str,
    test_name: Optional[str] = None
) -> str:
    """
    Retrieve recent lab and test results for a patient.
    Use this when a patient asks about their test results.

    Design rationale:
    This tool retrieves STRUCTURED numeric results from the lab system.
    Combined with the RAG memory (which understands the clinical CONTEXT
    of those results), the agent can provide informed, grounded responses
    about what results mean -- while staying within its authorized scope.
    """
    lab_results = {
        "P001": [
            {
                "test": "Fecal Calprotectin",
                "value": 450,
                "unit": "ug/g",
                "reference_range": "<50 ug/g",
                "status": "HIGH",
                "date": "2024-06-18",
                "ordering_physician": "Dr. House",
                "interpretation": "Elevated, suggests active intestinal inflammation"
            },
            {
                "test": "CRP (C-Reactive Protein)",
                "value": 18.2,
                "unit": "mg/L",
                "reference_range": "<5 mg/L",
                "status": "HIGH",
                "date": "2024-06-18",
                "ordering_physician": "Dr. House",
                "interpretation": "Elevated inflammatory marker"
            },
            {
                "test": "Hemoglobin",
                "value": 11.8,
                "unit": "g/dL",
                "reference_range": "13.5-17.5 g/dL",
                "status": "LOW",
                "date": "2024-06-18",
                "ordering_physician": "Dr. House",
                "interpretation": "Mild anemia, consistent with chronic inflammation"
            },
        ],
        "P002": [
            {
                "test": "Complete Blood Count",
                "value": "Normal",
                "unit": "",
                "reference_range": "Normal",
                "status": "NORMAL",
                "date": "2024-01-08",
                "ordering_physician": "Dr. House",
                "interpretation": "All values within normal limits"
            },
        ],
        "P003": [
            {
                "test": "Colonoscopy Pathology",
                "value": "Tubular adenoma x2, low grade dysplasia",
                "unit": "",
                "reference_range": "No polyps",
                "status": "ABNORMAL",
                "date": "2023-11-05",
                "ordering_physician": "Dr. House",
                "interpretation": "Low-risk adenomas, completely resected"
            },
        ]
    }

    results = lab_results.get(patient_id, [])

    if test_name:
        results = [r for r in results if test_name.lower() in r["test"].lower()]

    if not results:
        return json.dumps({
            "patient_id": patient_id,
            "results": [],
            "message": "No results found"
        })

    print(f"\n  [LabTool] Retrieved {len(results)} results for {patient_id}")
    for r in results:
        print(f"  {r['test']}: {r['value']} {r['unit']} [{r['status']}]")

    return json.dumps({
        "patient_id": patient_id,
        "results": results,
        "retrieved_at": datetime.now().isoformat()
    })


# ── TOOL REGISTRY ─────────────────────────────────────────────────────────────
# Design rationale:
# The tool registry is what gets passed to the LLM.
# The LLM sees the tool names, descriptions, and schemas.
# It decides autonomously which tools to call based on the conversation.
# This is function calling / tool use in action.

ALL_TOOLS = [
    send_patient_email,
    transcribe_voice_message,
    lookup_emr_data,
    manage_appointment,
    get_lab_results,
]

TOOL_NAMES = {tool.name: tool for tool in ALL_TOOLS}


if __name__ == "__main__":
    print("Testing tools...\n")

    # Test 1: Email
    print("=== EMAIL TOOL ===")
    result = send_patient_email.invoke({
        "to_patient_id": "P001",
        "subject": "Your appointment reminder",
        "body": "Dear John, your appointment is on September 15th.",
        "cc_physician": False
    })
    print(f"Result: {result}\n")

    # Test 2: EMR Lookup
    print("=== EMR LOOKUP TOOL ===")
    result = lookup_emr_data.invoke({
        "patient_id": "P001",
        "query_type": "medications"
    })
    print(f"Result: {result}\n")

    # Test 3: Lab Results
    print("=== LAB RESULTS TOOL ===")
    result = get_lab_results.invoke({
        "patient_id": "P001",
        "test_name": "Calprotectin"
    })
    print(f"Result: {result}\n")

    # Test 4: Appointment
    print("=== APPOINTMENT TOOL ===")
    result = manage_appointment.invoke({
        "patient_id": "P001",
        "action": "check_next"
    })
    print(f"Result: {result}\n")

    # Test 5: Phone transcription
    print("=== PHONE TRANSCRIPTION TOOL ===")
    result = transcribe_voice_message.invoke({
        "audio_file_path": "voicemail_001.mp3",
        "language": "en"
    })
    print(f"Result: {result}\n")

    print("All tools working correctly.")
