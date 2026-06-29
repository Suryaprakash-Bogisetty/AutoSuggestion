

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
    "summary":         "Continue with a concise summary of the patient's condition and key findings.",
    "clinical_notes":  "Continue with detailed clinical observations, history, and examination findings.",
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
    "summary":         "Summary",
    "clinical_notes":  "Clinical Notes",
}

REPHRASE_PURPOSE_HINTS = {
    "chief_complaint": "Rephrase to describe the symptom, body location, onset, and duration using clinical terminology.",
    "diagnosis":       "Rephrase using standard disease names and ICD-compatible terminology.",
    "investigations":  "Rephrase using full names of lab tests, imaging studies, and diagnostic procedures.",
    "medications":     "Rephrase using generic drug names with standard dosage format. Do not alter drug names.",
    "procedures":      "Rephrase using precise clinical or surgical procedure terminology.",
    "vitals":          "Rephrase to state each vital sign clearly with appropriate labels and units.",
    "advice_followup": "Rephrase as clear, formal patient instructions with specific timelines.",
    "doctors_notes":   "Rephrase as a structured clinical note with formal observations, impressions, and management plan.",
    "summary":         "Rephrase as a concise, formal medical summary highlighting key findings.",
    "clinical_notes":  "Rephrase as a structured and professional clinical note, detailing history and exam findings.",
}

REPHRASE_SYSTEM_PROMPT = """You are a clinical documentation specialist. Your task is to rephrase medical notes written by doctors into formal, accurate, and concise clinical text.

Rules:
1. Make the text concise and clinically formal.
2. Fix grammar, punctuation, sentence structure, and awkward phrasing.
3. Replace colloquial language with standard medical terminology.
4. Preserve all numeric values exactly — the number must not change. You may improve unit formatting (e.g. 325mg → 325 mg, 101.4F → 101.4°F) but never alter the value itself.
5. Do not add medical facts absent from the original — especially do not invent units (e.g. do not add ng/mL if troponin units were not stated).
6. Do not remove any existing clinical information.
7. Preserve drug names exactly (e.g. metformin stays metformin, not biguanide).
8. If the input is already written in proper clinical language, return it with only minimal corrections.
9. Do not explain your reasoning or add any commentary.
10. Return only the rephrased text — nothing else.

Examples:
Input: patient has high BP and sugar problems
Output: Patient presents with hypertension and diabetes mellitus.

Input: heart beating too fast, told him to rest
Output: Tachycardia noted. Rest advised.

Input: patient came in saying chest hurts and hard to breathe
Output: Patient presents with chest pain and dyspnea.

Input: gave him metformin 500mg two times daily for sugar
Output: Metformin 500mg twice daily prescribed for diabetes mellitus.

Input: Echo and blood tests ordered
Output: Echocardiography and hematological investigations ordered.

Input: BP 140/90 mmHg, SpO2 98%, HR 88/min, RR 18/min
Output: Blood pressure 140/90 mmHg, oxygen saturation 98%, heart rate 88/min, respiratory rate 18/min.

Input: Patient is a 45-year-old male with known hypertension presenting with headache since morning.
Output: Patient is a 45-year-old male with known hypertension presenting with headache since morning."""


def build_rephrase_user_message(text: str, purpose: str | None, context: dict) -> str:
    hint = REPHRASE_PURPOSE_HINTS.get(purpose, "") if purpose else ""
    label = _CONTEXT_LABELS.get(purpose, purpose.replace("_", " ").title()) if purpose else ""

    section_line = f"Section: {label}" + (f" — {hint}" if hint else "") + "\n" if label else ""

    other_lines = (
        "\n".join(
            f"  {_CONTEXT_LABELS.get(k, k.replace('_', ' ').title())}: {v}"
            for k, v in context.items()
            if k != purpose and v and str(v).strip()
        )
        if context
        else ""
    )
    context_block = f"\nOther Note Sections (for consistency):\n{other_lines}\n" if other_lines else ""

    return (
        f"{section_line}"
        f"{context_block}\n"
        f"Text to rephrase:\n{text}\n\n"
        f"Rephrased text:"
    )


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
