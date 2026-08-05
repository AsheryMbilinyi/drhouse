"""
agents/clinical_agents.py

MULTI-AGENT SYSTEM -- The reasoning core of DrHouse.

Key concepts:
- Each agent has a single, well-defined responsibility (single responsibility principle)
- Agents communicate via a shared typed state (LangGraph StateGraph)
- The planning engine decides which agents to invoke and in what order
- The reasoning module synthesizes EMR context with the patient message
- Tool-calling pipelines connect agents to external systems (EMR, email)

Agent pipeline:
  Patient Message
       ↓
  IntakeAgent      -- classifies message, identifies patient, extracts intent
       ↓
  MemoryAgent      -- retrieves relevant EMR context via RAG
       ↓
  ReasoningAgent   -- synthesizes context + message → appropriate response
       ↓
  GuardrailLayer   -- safety and compliance checks
       ↓
  [HITL checkpoint if required]
       ↓
  ResponseAgent    -- formats and sends the final response
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory.emr_memory import EMRMemorySystem, SYNTHETIC_EMR_NOTES
from guardrails.clinical_guardrails import ClinicalGuardrails


# ── TYPED STATE ───────────────────────────────────────────────────────────────
# Design rationale:
# Typed state is the shared memory between agents in the pipeline.
# Each agent reads from and writes to this state.
# TypedDict enforces the schema -- no agent can write unexpected fields.
# This is critical for debuggability in production systems.

class ClinicalAgentState(TypedDict):
    # Input
    patient_message: str
    patient_id: str
    communication_channel: str  # "email" or "phone"

    # Populated by IntakeAgent
    intent: str          # "symptom_report", "medication_question", "appointment_request", "test_results"
    urgency: str         # "routine", "soon", "urgent", "emergency"

    # Populated by MemoryAgent
    emr_context: str     # Retrieved relevant EMR chunks

    # Populated by ReasoningAgent
    draft_response: str  # Draft response before guardrail check

    # Populated by GuardrailLayer
    guardrail_passed: bool
    guardrail_reason: str
    requires_physician_review: bool
    risk_level: str

    # Populated by ResponseAgent
    final_response: str
    escalation_note: str  # Note to physician if escalation needed


# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── AGENT 1: INTAKE AGENT ────────────────────────────────────────────────────
def intake_agent(state: ClinicalAgentState) -> ClinicalAgentState:
    """
    Classifies the incoming message and extracts structured intent.

    Design rationale:
    The intake agent is the perception layer -- it transforms raw unstructured
    patient input into structured data the downstream agents can reason over.
    This is the 'perception module' in the agentic components framework.
    """
    print(f"\n[IntakeAgent] Processing message from {state['patient_id']}...")

    system_prompt = """You are a medical office intake classifier for a gastroenterology clinic.

Analyze the patient message and respond with ONLY a JSON object:
{
    "intent": "<one of: symptom_report, medication_question, appointment_request, test_results, general_inquiry>",
    "urgency": "<one of: routine, soon, urgent, emergency>",
    "key_topics": ["<list of main topics mentioned>"]
}

Urgency guidelines:
- emergency: chest pain, difficulty breathing, vomiting blood, loss of consciousness
- urgent: severe pain, blood in stool, high fever, inability to eat/drink
- soon: worsening symptoms, medication side effects, abnormal test results
- routine: appointment requests, general questions, mild symptoms"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Patient message: {state['patient_message']}")
    ])

    import json
    try:
        parsed = json.loads(response.content)
        intent = parsed.get("intent", "general_inquiry")
        urgency = parsed.get("urgency", "routine")
    except:
        intent = "general_inquiry"
        urgency = "routine"

    print(f"[IntakeAgent] Intent: {intent}, Urgency: {urgency}")

    return {**state, "intent": intent, "urgency": urgency}


# ── AGENT 2: MEMORY AGENT ────────────────────────────────────────────────────
def memory_agent(state: ClinicalAgentState, memory: EMRMemorySystem) -> ClinicalAgentState:
    """
    Retrieves relevant EMR context for the current patient and query.

    Design rationale:
    This is the long-term memory retrieval module. It uses semantic search
    (RAG) to find the most relevant patient history for the current query.
    Key design decision: we retrieve ONLY relevant chunks, not the full
    10-year history -- this keeps the LLM context window manageable and
    focuses the reasoning agent on what matters.
    """
    print(f"\n[MemoryAgent] Retrieving EMR context for patient {state['patient_id']}...")

    # Build a rich query from the message + intent
    retrieval_query = f"{state['patient_message']} {state['intent']}"

    results = memory.retrieve(
        query=retrieval_query,
        patient_id=state['patient_id'],
        k=4
    )

    if not results:
        emr_context = "No previous records found for this patient."
    else:
        context_parts = []
        for doc in results:
            context_parts.append(
                f"[{doc.metadata['date']} | {doc.metadata['type']}]\n{doc.page_content}"
            )
        emr_context = "\n\n---\n\n".join(context_parts)

    print(f"[MemoryAgent] Retrieved {len(results)} relevant chunks")

    return {**state, "emr_context": emr_context}


# ── AGENT 3: REASONING AGENT ─────────────────────────────────────────────────
def reasoning_agent(state: ClinicalAgentState) -> ClinicalAgentState:
    """
    Synthesizes EMR context with the patient message to generate a response.

    Design rationale:
    This is the reasoning module -- the cognitive core of the system.
    It uses chain-of-thought reasoning to:
    1. Understand what the patient is asking
    2. Ground its response in retrieved EMR context
    3. Decide what it can answer vs. what needs physician review
    4. Draft an appropriate, safe response

    Critically: the reasoning agent is INSTRUCTED to stay within scope.
    It cannot prescribe, diagnose, or override physician decisions.
    """
    print(f"\n[ReasoningAgent] Generating response draft...")

    system_prompt = f"""You are a clinical communications assistant for Pacific Digestive Health, 
a gastroenterology clinic. You assist Dr. House and the team by responding to patient 
messages based on their medical history.

STRICT RULES:
1. NEVER prescribe medications or change medication doses
2. NEVER provide a new diagnosis
3. ALWAYS refer clinical decisions to the physician
4. ONLY relay information that is explicitly in the patient's EMR
5. For urgent symptoms, acknowledge and escalate immediately
6. Be empathetic, clear, and professional

PATIENT CONTEXT FROM EMR:
{state['emr_context']}

PATIENT INTENT: {state['intent']}
URGENCY LEVEL: {state['urgency']}

Generate a draft response to the patient. If the matter requires physician review,
clearly state that the physician will follow up. Do not attempt to answer clinical
questions beyond what is documented in the EMR."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Patient message: {state['patient_message']}")
    ])

    print(f"[ReasoningAgent] Draft generated ({len(response.content)} chars)")

    return {**state, "draft_response": response.content}


# ── GUARDRAIL NODE ────────────────────────────────────────────────────────────
def guardrail_node(state: ClinicalAgentState) -> ClinicalAgentState:
    """
    Runs all guardrail checks on the draft response.

    Design rationale:
    Guardrails are a separate layer from the reasoning agent.
    They run after the agent generates a response but before it is sent.
    This separation means guardrails can be updated independently of agents.
    """
    print(f"\n[Guardrails] Running safety checks...")

    guardrails = ClinicalGuardrails()
    result = guardrails.check_all(
        agent_response=state['draft_response'],
        patient_message=state['patient_message'],
        patient_id=state['patient_id']
    )

    print(f"[Guardrails] Passed: {result.passed}, Risk: {result.risk_level}, Physician review: {result.requires_physician_review}")

    return {
        **state,
        "guardrail_passed": result.passed,
        "guardrail_reason": result.reason,
        "requires_physician_review": result.requires_physician_review,
        "risk_level": result.risk_level,
    }


# ── ROUTING LOGIC ─────────────────────────────────────────────────────────────
def route_after_guardrails(state: ClinicalAgentState) -> Literal["response_agent", "escalation_agent", "safety_block"]:
    """
    Planning engine -- decides what happens after guardrail checks.

    Design rationale:
    This is the planning module. Based on guardrail results and urgency,
    it routes to the appropriate next step. This is where the system
    makes autonomous decisions about workflow routing.
    """
    if not state['guardrail_passed']:
        return "safety_block"
    elif state['requires_physician_review'] or state['urgency'] in ['urgent', 'emergency']:
        return "escalation_agent"
    else:
        return "response_agent"


# ── AGENT 4: RESPONSE AGENT ──────────────────────────────────────────────────
def response_agent(state: ClinicalAgentState) -> ClinicalAgentState:
    """Formats and finalizes the response for sending."""
    print(f"\n[ResponseAgent] Finalizing response...")

    final_response = f"""Dear {state['patient_id']},

{state['draft_response']}

If you have any urgent concerns, please call our clinic directly at (604) 555-0100.

Pacific Digestive Health Team"""

    return {**state, "final_response": final_response, "escalation_note": ""}


# ── ESCALATION AGENT ─────────────────────────────────────────────────────────
def escalation_agent(state: ClinicalAgentState) -> ClinicalAgentState:
    """
    Handles cases requiring physician review.

    Design rationale:
    This is the human-in-the-loop checkpoint. When triggered, the agent:
    1. Sends the patient an acknowledgment
    2. Creates a physician alert with full context
    3. Does NOT attempt to answer the clinical question autonomously
    """
    print(f"\n[EscalationAgent] Escalating to physician review...")

    patient_response = f"""Dear {state['patient_id']},

Thank you for your message. We have received your inquiry and a member of our 
medical team will review and respond to you within 24 hours.

If you are experiencing a medical emergency, please call 911 or go to your 
nearest emergency department immediately.

Pacific Digestive Health Team"""

    physician_note = f"""
PHYSICIAN ALERT -- Patient {state['patient_id']} requires review

Urgency: {state['urgency']}
Guardrail flag: {state['guardrail_reason']}

Patient message:
{state['patient_message']}

Relevant EMR context:
{state['emr_context'][:500]}...

Suggested action: Review patient message and respond directly.
"""

    return {
        **state,
        "final_response": patient_response,
        "escalation_note": physician_note
    }


# ── SAFETY BLOCK ─────────────────────────────────────────────────────────────
def safety_block(state: ClinicalAgentState) -> ClinicalAgentState:
    """Blocks the response and alerts the team."""
    print(f"\n[SafetyBlock] Response blocked: {state['guardrail_reason']}")

    return {
        **state,
        "final_response": "This response was blocked by the safety system and requires manual review.",
        "escalation_note": f"SAFETY BLOCK: {state['guardrail_reason']}\n\nBlocked response:\n{state['draft_response']}"
    }


# ── BUILD THE GRAPH ───────────────────────────────────────────────────────────
def build_agent_graph(memory: EMRMemorySystem) -> StateGraph:
    """
    Assembles the full agent graph.

    Design rationale:
    LangGraph represents the agent pipeline as a directed graph where:
    - Nodes are agent functions
    - Edges are the flow between agents
    - Conditional edges implement the planning/routing logic
    - MemorySaver enables short-term memory across conversation turns
    """
    graph = StateGraph(ClinicalAgentState)

    # Add nodes
    graph.add_node("intake_agent", intake_agent)
    graph.add_node("memory_agent", lambda s: memory_agent(s, memory))
    graph.add_node("reasoning_agent", reasoning_agent)
    graph.add_node("guardrail_node", guardrail_node)
    graph.add_node("response_agent", response_agent)
    graph.add_node("escalation_agent", escalation_agent)
    graph.add_node("safety_block", safety_block)

    # Define flow
    graph.set_entry_point("intake_agent")
    graph.add_edge("intake_agent", "memory_agent")
    graph.add_edge("memory_agent", "reasoning_agent")
    graph.add_edge("reasoning_agent", "guardrail_node")

    # Conditional routing after guardrails
    graph.add_conditional_edges(
        "guardrail_node",
        route_after_guardrails,
        {
            "response_agent": "response_agent",
            "escalation_agent": "escalation_agent",
            "safety_block": "safety_block",
        }
    )

    graph.add_edge("response_agent", END)
    graph.add_edge("escalation_agent", END)
    graph.add_edge("safety_block", END)

    # MemorySaver enables short-term memory (conversation history)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
