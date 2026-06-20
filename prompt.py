import json

SYSTEM_PROMPT = (
    "You are a medical documentation assistant that auto-completes clinical notes. "
    "Continue the given prefix naturally based on the Clinical Note. "
    "Use the Context as supporting hints only. "
    "IMPORTANT: Add NEW clinical detail — such as duration, onset, severity, location, or associated findings — "
    "that is NOT already stated in the Clinical Note. Do not restate or rephrase what is already written. "
    "Return ONLY the completion words — no punctuation, no explanation, no repetition of the prefix.\n\n"
    "Example:\n"
    "  Purpose: chief_complaint | Prefix: \"patient complains of\"\n"
    "  Clinical Note: he came with fever and joint pain\n"
    "  Context — Chief Complaint: fever and joint pain since 3 days\n"
    "  BAD Completion: fever and joint pain  ← already in the note, adds nothing\n"
    "  GOOD Completion: joint pain with fever since 3 days"
)

PURPOSE_HINTS = {
    "chief_complaint": "Focus on the patient's primary symptom or reason for visit.",
    "diagnosis":       "Focus on disease or condition names relevant to the note.",
    "investigations":  "Focus on lab tests, imaging, or diagnostic procedures.",
    "medications":     "Focus on drug names, dosage, or treatment regimens.",
    "procedures":      "Focus on clinical or surgical procedure names.",
    "vitals":          "Focus on vital sign values such as temperature, pulse, BP, SpO2, or respiratory rate.",
    "advice_followup": "Focus on patient advice, lifestyle instructions, or follow-up schedule.",
    "doctors_notes":   "Focus on clinical observations, impressions, or treatment plan notes by the doctor.",
}

_CONTEXT_LABELS = {
    "chief_complaint": "Chief Complaint",
    "diagnosis":       "Diagnosis",
    "investigations":  "Investigations",
    "medications":     "Medications",
    "procedures":      "Procedures",
    "vitals":          "Vitals",
    "advice_followup": "Advice & Follow-up",
    "doctors_notes":   "Doctor's Notes",
}


def build_user_message(prefix: str, full_note: str, context: dict, purpose: str) -> str:
    hint = PURPOSE_HINTS.get(purpose, "")
    context_lines = "\n".join(
        f"  {_CONTEXT_LABELS.get(k, k.replace('_', ' ').title())}: {v}"
        for k, v in context.items()
        if v and str(v).strip()
    ) or "  No context provided."
    return (
        f"Purpose: {purpose} — {hint}\n\n"
        f"Clinical Note:\n{full_note}\n\n"
        f"Context (hints):\n{context_lines}\n\n"
        f'Continue the prefix for the "{purpose}" section:\n'
        f'Prefix: "{prefix}"\n'
        f"Completion:"
    )
