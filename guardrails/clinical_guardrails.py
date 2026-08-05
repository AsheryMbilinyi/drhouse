"""
guardrails/clinical_guardrails.py

GUARDRAIL LAYER -- Privacy, safety, and compliance checks.

Key concepts:
- Guardrails are SEPARATE from the agent's reasoning
- They run AFTER the agent generates a response, BEFORE it is sent
- They catch: PII leakage, clinical overreach, hallucinated medications,
  responses that should be escalated to a physician
- In a regulated environment (PHIPA/PIPEDA), guardrails are non-negotiable

Design rationale:
"I separate guardrails from evaluators. Guardrails block bad outputs
at serving time. Evaluators measure quality offline. Mixing them
creates feedback loops that corrupt your quality signal."
"""

from pydantic import BaseModel
from typing import Tuple
import re


class GuardrailResult(BaseModel):
    passed: bool
    reason: str
    requires_physician_review: bool = False
    risk_level: str = "low"  # low, medium, high, critical


class ClinicalGuardrails:
    """
    Multi-layer guardrail system for clinical communications.

    Layers (in order of execution):
    1. PII check -- no unauthorized personal data in response
    2. Clinical safety check -- no dangerous medical advice
    3. Scope check -- agent stays within its authorized boundaries
    4. Escalation check -- flags cases that need physician review
    """

    # Medications that should NEVER be recommended without physician review
    HIGH_RISK_MEDICATIONS = [
        "prednisone", "methotrexate", "azathioprine", "infliximab",
        "adalimumab", "vedolizumab", "ustekinumab", "tofacitinib",
        "opioid", "morphine", "hydrocodone", "oxycodone",
        "warfarin", "heparin", "insulin"
    ]

    # Symptoms that require urgent escalation
    URGENT_SYMPTOMS = [
        "blood in stool", "rectal bleeding", "severe abdominal pain",
        "cannot eat", "cannot drink", "high fever", "chest pain",
        "difficulty breathing", "fainting", "unconscious",
        "vomiting blood", "black tarry stool"
    ]

    # Things the agent should NEVER do
    FORBIDDEN_ACTIONS = [
        "prescribe", "diagnosis confirmed", "you definitely have",
        "stop all medications", "you don't need to see a doctor",
        "ignore your symptoms", "this is not serious"
    ]

    def check_all(
        self,
        agent_response: str,
        patient_message: str,
        patient_id: str
    ) -> GuardrailResult:
        """
        Run all guardrail checks in sequence.
        Fail fast -- return on first critical failure.
        """

        # Layer 1: PII check
        pii_result = self._check_pii_leakage(agent_response, patient_id)
        if not pii_result.passed:
            return pii_result

        # Layer 2: Forbidden actions
        forbidden_result = self._check_forbidden_actions(agent_response)
        if not forbidden_result.passed:
            return forbidden_result

        # Layer 3: Clinical safety
        safety_result = self._check_clinical_safety(
            agent_response, patient_message
        )
        if not safety_result.passed:
            return safety_result

        # Layer 4: Escalation check (non-blocking -- flags for review)
        escalation_result = self._check_escalation_needed(
            agent_response, patient_message
        )

        return escalation_result

    def _check_pii_leakage(
        self, response: str, current_patient_id: str
    ) -> GuardrailResult:
        """
        Check that the response does not leak other patients' data.

        Design rationale:
        In a multi-patient system, we must ensure the agent never
        cross-contaminates patient data. This is a PHIPA/HIPAA requirement.
        """
        # Check for other patient IDs in the response
        patient_ids_found = re.findall(r'P\d{3}', response)
        other_patients = [
            pid for pid in patient_ids_found
            if pid != current_patient_id
        ]

        if other_patients:
            return GuardrailResult(
                passed=False,
                reason=f"PII VIOLATION: Response contains data from other patients: {other_patients}",
                requires_physician_review=False,
                risk_level="critical"
            )

        return GuardrailResult(passed=True, reason="PII check passed")

    def _check_forbidden_actions(self, response: str) -> GuardrailResult:
        """
        Check that the agent does not take actions outside its scope.
        """
        response_lower = response.lower()

        for forbidden in self.FORBIDDEN_ACTIONS:
            if forbidden in response_lower:
                return GuardrailResult(
                    passed=False,
                    reason=f"SCOPE VIOLATION: Response contains forbidden phrase: '{forbidden}'",
                    requires_physician_review=True,
                    risk_level="high"
                )

        return GuardrailResult(passed=True, reason="Scope check passed")

    def _check_clinical_safety(
        self, response: str, patient_message: str
    ) -> GuardrailResult:
        """
        Check for unsafe clinical recommendations.

        Design rationale:
        The agent can provide general information and relay physician
        instructions from the EMR. It cannot recommend new medications
        or change existing medication doses without physician approval.
        """
        response_lower = response.lower()

        # Check if agent is recommending high-risk medications unprompted
        for med in self.HIGH_RISK_MEDICATIONS:
            if med in response_lower:
                # Only flag if this is a NEW recommendation, not relaying
                # existing physician instructions
                if any(phrase in response_lower for phrase in [
                    "i recommend", "you should take", "start taking",
                    "i suggest taking", "try taking"
                ]):
                    return GuardrailResult(
                        passed=False,
                        reason=f"CLINICAL SAFETY: Agent attempting to recommend medication '{med}' without physician authorization",
                        requires_physician_review=True,
                        risk_level="critical"
                    )

        return GuardrailResult(passed=True, reason="Clinical safety check passed")

    def _check_escalation_needed(
        self, response: str, patient_message: str
    ) -> GuardrailResult:
        """
        Flag cases that need physician review even if response is otherwise safe.
        Non-blocking -- the response can proceed but is flagged for review.

        Design rationale:
        This is the human-in-the-loop checkpoint. Not all messages need
        physician review, but urgent symptoms always do. The agent
        acknowledges the message and tells the patient a physician will
        respond, rather than attempting to handle it autonomously.
        """
        message_lower = patient_message.lower()

        for symptom in self.URGENT_SYMPTOMS:
            if symptom in message_lower:
                return GuardrailResult(
                    passed=True,  # Does not block -- but flags
                    reason=f"ESCALATION REQUIRED: Urgent symptom detected: '{symptom}'",
                    requires_physician_review=True,
                    risk_level="high"
                )

        return GuardrailResult(
            passed=True,
            reason="All guardrail checks passed",
            requires_physician_review=False,
            risk_level="low"
        )


if __name__ == "__main__":
    guardrails = ClinicalGuardrails()

    # Test 1: Safe response
    result = guardrails.check_all(
        agent_response="Thank you for your message. Dr. House will review your calprotectin results and respond shortly.",
        patient_message="I got my test results back",
        patient_id="P001"
    )
    print(f"Test 1 (safe): passed={result.passed}, risk={result.risk_level}")
    print(f"  Reason: {result.reason}\n")

    # Test 2: Urgent symptom
    result = guardrails.check_all(
        agent_response="I understand you are concerned. Let me check your records.",
        patient_message="I have blood in stool and severe abdominal pain",
        patient_id="P001"
    )
    print(f"Test 2 (urgent): passed={result.passed}, physician_review={result.requires_physician_review}")
    print(f"  Reason: {result.reason}\n")

    # Test 3: Forbidden action
    result = guardrails.check_all(
        agent_response="Based on your symptoms, diagnosis confirmed as IBS. You don't need to see a doctor.",
        patient_message="I have stomach pain",
        patient_id="P002"
    )
    print(f"Test 3 (forbidden): passed={result.passed}, risk={result.risk_level}")
    print(f"  Reason: {result.reason}")
