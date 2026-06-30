LANGUAGE_MAP: dict[str, str] = {
    "te": "Telugu",
    "hi": "Hindi",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "bn": "Bengali",
    "pa": "Punjabi",
    "gu": "Gujarati",
    "or": "Odia",
    "as": "Assamese",
}


def build_system_prompt(language_name: str, domain: str) -> str:
    base = (
        f"You are a professional translator specializing in Indian languages.\n"
        f"Translate the given English text to {language_name} using native {language_name} script.\n\n"
        f"Rules:\n"
        f"1. Output ONLY the translated text — no explanation, no prefix like 'Translation:', no commentary.\n"
        f"2. Translate conversational/descriptive words and sentences into {language_name}.\n"
        f"3. Use formal, standard language register.\n"
        f"4. Do not add any information not present in the original text.\n"
        f"5. Preserve all numbers, measurements, and units exactly as given (e.g., 500mg, 145/92, 7.8%).\n"
        f"6. The following medical terms must stay EXACTLY in English — do NOT translate or transliterate them into {language_name} script:\n"
        f"   a) Drug and medication names (e.g., Metformin, Ibuprofen, Lisinopril, Amlodipine, Aspirin, Warfarin)\n"
        f"   b) Disease names and diagnoses (e.g., Hypertension, Diabetes Mellitus, Migraine, Asthma, COPD, Hypothyroidism)\n"
        f"   c) Medical disorders and conditions (e.g., Atrial Fibrillation, Tachycardia, Anemia, Sepsis)\n"
        f"   d) Medical and surgical procedures (e.g., Angioplasty, Appendectomy, Coronary Artery Bypass Graft, Biopsy)\n"
        f"   e) Laboratory tests and investigations (e.g., CBC, HbA1c, ECG, MRI, CT Scan, LFT, RFT, Troponin, INR, SpO2)\n"
        f"   f) Medical abbreviations (BP, HR, RR, BMI, eGFR, IV, OD, BD, TDS, stat)\n"
        f"   — These terms must appear in the output exactly as written in the input (Latin/English script).\n"
    )
    if domain == "medical":
        base += (
            f"7. Dosage instructions must stay in English exactly as written "
            f"(e.g., '500mg twice daily', '20 units subcutaneously' stay unchanged).\n"
        )
    return base


def build_user_message(text: str, domain: str, context: dict | None) -> str:
    context_block = ""
    if domain == "medical" and context:
        parts = []
        if context.get("diagnoses"):
            val = context["diagnoses"]
            parts.append(f"Diagnoses: {', '.join(val) if isinstance(val, list) else val}")
        if context.get("medications"):
            val = context["medications"]
            parts.append(f"Medications: {', '.join(val) if isinstance(val, list) else val}")
        if context.get("chief_complaint"):
            parts.append(f"Chief Complaint: {context['chief_complaint']}")
        if context.get("vitals"):
            parts.append(f"Vitals: {context['vitals']}")
        if parts:
            context_block = (
                "Patient Context (reference only — do not translate these terms):\n"
                + "\n".join(parts)
                + "\n\n"
            )
    return f"{context_block}Text to translate:\n{text}"
