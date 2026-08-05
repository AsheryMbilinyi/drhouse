"""
main.py -- DrHouse entry point

Demonstrates the full multi-agent pipeline with:
- Memory system (RAG-based EMR retrieval)
- Tool-calling (email, EMR lookup, lab results, appointments, voice)
- Planning engine (conditional routing)
- Guardrails (safety and compliance)
- Human-in-the-loop (physician escalation)

Usage:
    export OPENAI_API_KEY=your_key_here
    python main.py
"""

import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory.emr_memory import EMRMemorySystem, SYNTHETIC_EMR_NOTES
from agents.clinical_agents import build_agent_graph
from tools.clinical_tools import (
    lookup_emr_data, get_lab_results,
    manage_appointment, send_patient_email,
    transcribe_voice_message, ALL_TOOLS
)

BLUE   = "\033[94m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"


def run_scenario(graph, scenario, thread_id):
    print(f"\n{'='*65}")
    print(f"{BOLD}{BLUE}SCENARIO: {scenario['name']}{RESET}")
    print(f"  Patient: {scenario['patient_id']}")
    print(f"  Channel: {scenario.get('channel', 'email')}")
    print(f"  Message: {scenario['message'][:80]}")
    print(f"{'='*65}")

    # Handle voice input
    if scenario.get('channel') == 'phone':
        print(f"\n{CYAN}[Voice] Transcribing voicemail...{RESET}")
        result = transcribe_voice_message.invoke({
            "audio_file_path": scenario.get('audio_file', 'voicemail_001.mp3'),
            "language": "en"
        })
        message = json.loads(result).get('text', scenario['message'])
    else:
        message = scenario['message']

    # Pre-fetch structured tool data if needed
    if scenario.get('prefetch_tools'):
        print(f"\n{CYAN}[Tools] Pre-fetching structured data...{RESET}")
        for tc in scenario['prefetch_tools']:
            if tc['tool'] == 'lab_results':
                get_lab_results.invoke({
                    "patient_id": scenario['patient_id'],
                    "test_name": tc.get('test_name')
                })
            elif tc['tool'] == 'appointment':
                manage_appointment.invoke({
                    "patient_id": scenario['patient_id'],
                    "action": "check_next"
                })

    initial_state = {
        "patient_message": message,
        "patient_id": scenario['patient_id'],
        "communication_channel": scenario.get('channel', 'email'),
        "intent": "", "urgency": "", "emr_context": "",
        "draft_response": "", "guardrail_passed": False,
        "guardrail_reason": "", "requires_physician_review": False,
        "risk_level": "low", "final_response": "", "escalation_note": "",
    }

    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(initial_state, config=config)

    print(f"\n{BOLD}--- PIPELINE SUMMARY ---{RESET}")
    print(f"  Intent:           {result.get('intent')}")
    print(f"  Urgency:          {result.get('urgency')}")
    print(f"  Guardrail passed: {result.get('guardrail_passed')}")
    print(f"  Risk level:       {result.get('risk_level')}")
    print(f"  Physician review: {result.get('requires_physician_review')}")

    print(f"\n{BOLD}--- RESPONSE TO PATIENT ---{RESET}")
    print(f"{GREEN}{result['final_response']}{RESET}")

    if result.get('escalation_note'):
        print(f"\n{BOLD}--- PHYSICIAN ALERT ---{RESET}")
        print(f"{YELLOW}{result['escalation_note']}{RESET}")
        send_patient_email.invoke({
            "to_patient_id": result['patient_id'],
            "subject": f"ALERT: Patient {result['patient_id']} requires review",
            "body": result['escalation_note'],
            "cc_physician": True
        })


def main():
    print(f"\n{BOLD}DrHouse -- Pacific Digestive Health{RESET}")

    print(f"\n{BLUE}Building EMR memory system...{RESET}")
    memory = EMRMemorySystem(persist_directory="./chroma_db")
    memory.build_memory(SYNTHETIC_EMR_NOTES)

    print(f"\n{BLUE}Assembling multi-agent graph...{RESET}")
    graph = build_agent_graph(memory)
    print(f"{GREEN}Ready. {len(ALL_TOOLS)} tools registered.{RESET}")

    scenarios = [
        {
            "name": "1. Routine medication question",
            "patient_id": "P001",
            "channel": "email",
            "message": "Hi, I wanted to check if I should still be taking the Budesonide. I have been on it for 6 weeks now.",
        },
        {
            "name": "2. Lab results inquiry with tool call",
            "patient_id": "P001",
            "channel": "email",
            "message": "My calprotectin came back at 850 ug/g. My GP said this is very high. Should I be worried?",
            "prefetch_tools": [{"tool": "lab_results", "test_name": "Calprotectin"}],
        },
        {
            "name": "3. Urgent symptom -- ESCALATION triggered",
            "patient_id": "P001",
            "channel": "email",
            "message": "I have blood in stool and severe abdominal pain since this morning.",
        },
        {
            "name": "4. Voice message -- PHONE channel",
            "patient_id": "P001",
            "channel": "phone",
            "audio_file": "voicemail_001.mp3",
            "message": "",
        },
        {
            "name": "5. Appointment inquiry with tool call",
            "patient_id": "P003",
            "channel": "email",
            "message": "Hello, I was told I need a follow-up colonoscopy. Can you remind me when it is scheduled?",
            "prefetch_tools": [{"tool": "appointment"}],
        },
    ]

    for i, scenario in enumerate(scenarios):
        run_scenario(graph, scenario, thread_id=f"session_{i}")
        if i < len(scenarios) - 1:
            input(f"\n{YELLOW}[Press Enter for next scenario]{RESET}")

    print(f"\n{BOLD}{GREEN}All scenarios complete.{RESET}")
    print(f"\n{CYAN}Components demonstrated:{RESET}")
    components = [
        "Memory systems      -- RAG retrieval from ChromaDB EMR vector store",
        "Planning engine     -- Conditional routing based on guardrails + urgency",
        "Reasoning modules   -- Context-grounded chain-of-thought with EMR data",
        "Tool-calling        -- Email, EMR lookup, lab results, appointments, voice",
        "Guardrails          -- PII, clinical safety, scope, escalation checks",
        "Human-in-the-loop   -- Physician escalation for urgent/unsafe cases",
        "Short-term memory   -- LangGraph MemorySaver across conversation turns",
        "Long-term memory    -- ChromaDB vector store for 10 years of EMR data",
    ]
    for c in components:
        print(f"  {c}")


if __name__ == "__main__":
    main()
