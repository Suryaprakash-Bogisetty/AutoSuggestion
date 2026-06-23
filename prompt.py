SYSTEM_PROMPT = ("""You are a medical note autocomplete assistant.

Your task is to predict the next words a doctor is most likely to type.

Use:
- the text already written
- the current note section
- the available clinical context

Rules:
1. Understand the entire note before generating a suggestion.
2. Predict the most likely continuation of the doctor's text.
3. Continue naturally from the current cursor position.
4. Use relevant clinical context when helpful.
5. Prefer likely continuation over introducing new information.
6. Maintain proper grammar and clinical documentation style.
7. Keep suggestions concise, typically 2-12 words.
8. Do not repeat text that appears immediately before the cursor.
9. Do not invent unsupported medical facts.
10. Do not explain your reasoning.

Return only the continuation text.""")
PURPOSE_HINTS = {
    "chief_complaint": "Continue with the symptom name, body location, onset, or duration.",
    "diagnosis":       "Continue with a disease or condition name supported by the note.",
    "investigations":  "Continue with a specific lab test, imaging study, or diagnostic procedure name.",
    "medications":     "Continue with a drug name and dosage only. Do not add time durations or instructions.",
    "procedures":      "Continue with a specific clinical or surgical procedure name.",
    "vitals":          "Continue with a numeric vital sign value such as temperature, BP, SpO2, or pulse.",
    "advice_followup": "Continue with a specific patient instruction or follow-up timeline.",
    "doctors_notes":   "Continue with a clinical observation, impression, or next step in the treatment plan.",
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


def build_user_message(text_before_cursor: str, context: dict, purpose: str) -> str:
    hint = PURPOSE_HINTS.get(purpose, "")
    label = _CONTEXT_LABELS.get(purpose, purpose.replace("_", " ").title())

    other_lines = "\n".join(
        f"  {_CONTEXT_LABELS.get(k, k.replace('_', ' ').title())}: {v}"
        for k, v in context.items()
        if k != purpose and v and str(v).strip()
    ) or "  No other fields filled."

    return (
        f"Purpose: {purpose} — {hint}\n\n"
        f"Other Fields (hints):\n{other_lines}\n\n"
        f"Text Before Cursor:\n{text_before_cursor}\n\n"
        f"Predict the next words:"
    )
