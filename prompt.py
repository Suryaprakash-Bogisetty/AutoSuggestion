import json

SYSTEM_PROMPT = (
    "You are a medical documentation assistant helping to auto-complete clinical notes. "
    "Your job is to complete a clinical sentence prefix with exactly 5-6 words based on the "
    "provided context and the purpose of the note section. "
    "Return ONLY the completion words — no punctuation, no explanation, no repetition of the prefix."
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


def build_user_message(prefix: str, full_note: str, context: dict, purpose: str) -> str:
    hint = PURPOSE_HINTS.get(purpose, "")
    return (
        f"Purpose: {purpose}\n"
        f"Guidance: {hint}\n\n"
        f"Context:\n{json.dumps(context, indent=2)}\n\n"
        f"Clinical Note:\n{full_note}\n\n"
        f'Complete the following prefix with 5-6 words for the "{purpose}" section:\n'
        f'Prefix: "{prefix}"\n'
        f"Completion:"
    )
