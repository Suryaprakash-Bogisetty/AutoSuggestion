"""
7B vs 72B rephrase comparison — latency, accuracy, quality.
Runs both models in parallel per case. Prints a side-by-side report.
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

from openai import AsyncOpenAI
from prompt import REPHRASE_SYSTEM_PROMPT, build_rephrase_user_message
from config import DEEPINFRA_API_KEY, DEEPINFRA_BASE_URL

client = AsyncOpenAI(api_key=DEEPINFRA_API_KEY, base_url=DEEPINFRA_BASE_URL)

MODEL_7B  = "Qwen/Qwen2-7B-Instruct"
MODEL_72B = "Qwen/Qwen2-72B-Instruct"

# ── Test cases ────────────────────────────────────────────────────────────────
CASES = [
    {
        "id": 1,
        "label": "Chief Complaint — heavy grammar errors + colloquial",
        "purpose": "chief_complaint",
        "context": {},
        "text": (
            "patient he came hospital yesterday night only, "
            "his chest is paining very much since 3 days and "
            "also breathe is not coming properly, sweating also "
            "there, he is saying pain is going to left arm also."
        ),
    },
    {
        "id": 2,
        "label": "Diagnosis — abbreviations + wrong spelling + informal",
        "purpose": "diagnosis",
        "context": {"chief_complaint": "chest pain radiating to left arm, sweating"},
        "text": (
            "looks like STEMI (inferio wall), also pt has DM type2 "
            "and HTN since 10 yrs. Kindey function also little bit affected. "
            "Previous histry of angioplasty 2 yrs back."
        ),
    },
    {
        "id": 3,
        "label": "Medications — dosage heavy, grammar broken",
        "purpose": "medications",
        "context": {"diagnosis": "Type 2 diabetes mellitus, hypertension"},
        "text": (
            "gave metformin 500mg two time daily, "
            "amlodipine 5mg one time morning, "
            "aspirin 75mg once daily at night, "
            "atorvastatin 40mg bedtime, "
            "ramipril 2.5mg morning empty stomch. "
            "also pantoprazole 40mg before food."
        ),
    },
    {
        "id": 4,
        "label": "Vitals — numbers + units + scattered format",
        "purpose": "vitals",
        "context": {},
        "text": (
            "BP 160 by 100 mmhg, pulse 98 per minute irregular, "
            "spo2 93 percentage on room air, "
            "temprature 100.4 fehrenheit, "
            "RR 22 per min, weight 87 kg, height 165 cm, "
            "random blood sugar 312 mg/dl."
        ),
    },
    {
        "id": 5,
        "label": "Investigations — mixed abbreviations + spelling errors",
        "purpose": "investigations",
        "context": {"diagnosis": "acute STEMI, CKD stage 3"},
        "text": (
            "ECG done showing ST elevaton in II III aVF leads. "
            "2D echo orderd. trop I came 4.8 which is high. "
            "CBC, LFT, RFT, serum electrolites sent. "
            "chest xray PA view taken. coronery angiograpy planned."
        ),
    },
    {
        "id": 6,
        "label": "Doctor's Notes — long paragraph, very informal, grammar disaster",
        "purpose": "doctors_notes",
        "context": {
            "chief_complaint": "chest pain, breathlessness",
            "diagnosis": "acute inferior STEMI, DM2, HTN, CKD3",
        },
        "text": (
            "65 year old male pt came to emergency with complain of chest pain "
            "since 2 day. pain is very sever and going to left arm. "
            "he is diabatic and have high BP from long time. kidney also week. "
            "on examination he look pale and sweating. BP was 158/96, "
            "pulse 104/min, spo2 94% on room air. "
            "ECG done and showing inferioir MI changes. "
            "troponin 4.8, creatinine 1.9 (high). "
            "we start him on aspirin 325mg, clopidogrel 300mg loading, "
            "heparin drip, statin and shifted to CCU. "
            "cardiology team informed and primary PCI is plan. "
            "family counsel done regardng prognosis and procedur risk."
        ),
    },
    {
        "id": 7,
        "label": "Advice & Follow-up — casual + Hindi-English mix",
        "purpose": "advice_followup",
        "context": {"diagnosis": "Type 2 diabetes, hypertension"},
        "text": (
            "patient ko bola ki medicine regularly lena hai, "
            "BP aur sugar monitor karte rehna. "
            "low salt aur low sugar khana khao. "
            "walking 30 min daily karo. "
            "ek hafte baad blood test karao — HbA1c, lipid profile, RFT. "
            "2 week me follow up aana."
        ),
    },
    {
        "id": 8,
        "label": "Already well-written — should return minimal changes",
        "purpose": "doctors_notes",
        "context": {},
        "text": (
            "Patient is a 52-year-old female with known hypertension and "
            "type 2 diabetes mellitus presenting with a 2-day history of "
            "progressive dyspnea on exertion and bilateral pedal edema. "
            "Examination reveals elevated JVP, bilateral basal crepitations, "
            "and S3 gallop. Chest X-ray shows cardiomegaly with pulmonary "
            "venous congestion. BNP is 1240 pg/mL. "
            "Impression: Acute decompensated heart failure."
        ),
    },
]

# ── Core call ─────────────────────────────────────────────────────────────────
async def call_model(model: str, text: str, purpose: str, context: dict) -> tuple[str, float]:
    msg = build_rephrase_user_message(text=text, purpose=purpose, context=context)
    t0 = time.monotonic()
    try:
        r = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": REPHRASE_SYSTEM_PROMPT},
                {"role": "user",   "content": msg},
            ],
            max_tokens=1000,
            temperature=0.05,
            top_p=0.85,
            frequency_penalty=0.1,
            timeout=90,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        output = (r.choices[0].message.content or "").strip()
    except Exception as e:
        output = f"[ERROR: {e}]"
    latency = time.monotonic() - t0
    return output, latency


# ── Run one case ──────────────────────────────────────────────────────────────
async def run_case(case: dict) -> dict:
    (out_7b, lat_7b), (out_72b, lat_72b) = await asyncio.gather(
        call_model(MODEL_7B,  case["text"], case["purpose"], case["context"]),
        call_model(MODEL_72B, case["text"], case["purpose"], case["context"]),
    )
    return {**case, "out_7b": out_7b, "lat_7b": lat_7b,
                    "out_72b": out_72b, "lat_72b": lat_72b}


# ── Scoring helpers ───────────────────────────────────────────────────────────
import re

def count_medical_terms(text: str) -> int:
    """Count presence of clinical/formal medical terms (simple heuristic)."""
    terms = [
        "presents", "presenting", "history", "examination", "assessment",
        "management", "impression", "prescribed", "administered", "initiated",
        "elevated", "noted", "revealed", "demonstrated", "indicated",
        "hypertension", "diabetes mellitus", "myocardial", "dyspnea",
        "diaphoresis", "tachycardia", "bradycardia", "creatinine",
        "troponin", "electrocardiogram", "echocardiography",
        "percutaneous", "coronary", "angiography", "reperfusion",
    ]
    t = text.lower()
    return sum(1 for w in terms if w in t)

def number_preservation(original: str, output: str) -> tuple[int, int]:
    orig_nums = {n for n in re.findall(r"\d+(?:\.\d+)?", original)
                 if float(n) >= 10 or "." in n}
    out_nums  = set(re.findall(r"\d+(?:\.\d+)?", output))
    preserved = orig_nums & out_nums
    return len(preserved), len(orig_nums)

def word_count(text: str) -> int:
    return len(text.split())

def has_structure(text: str) -> bool:
    """Check if output has section headers or bullet structure."""
    return bool(re.search(r"(\*\*|##|^\s*[-•]|\d+\.)", text, re.MULTILINE))


# ── Print report ──────────────────────────────────────────────────────────────
SEP  = "─" * 100
SEP2 = "═" * 100

def print_case(r: dict):
    print(f"\n{SEP2}")
    print(f"  CASE {r['id']}: {r['label']}")
    print(f"  Purpose: {r['purpose']}")
    print(SEP2)

    print(f"\n  INPUT ({word_count(r['text'])} words):")
    for line in r['text'].split(". "):
        print(f"    {line.strip()}")

    pres_7b,  total = number_preservation(r["text"], r["out_7b"])
    pres_72b, _     = number_preservation(r["text"], r["out_72b"])
    terms_7b  = count_medical_terms(r["out_7b"])
    terms_72b = count_medical_terms(r["out_72b"])

    print(f"\n  {'─'*45} 7B ({r['lat_7b']:.1f}s) {'─'*45}")
    print(f"  Words: {word_count(r['out_7b'])}  |  "
          f"Numbers preserved: {pres_7b}/{total}  |  "
          f"Clinical terms: {terms_7b}  |  "
          f"Structured: {has_structure(r['out_7b'])}")
    print()
    for line in r["out_7b"].split("\n"):
        if line.strip():
            print(f"  {line}")

    print(f"\n  {'─'*44} 72B ({r['lat_72b']:.1f}s) {'─'*44}")
    print(f"  Words: {word_count(r['out_72b'])}  |  "
          f"Numbers preserved: {pres_72b}/{total}  |  "
          f"Clinical terms: {terms_72b}  |  "
          f"Structured: {has_structure(r['out_72b'])}")
    print()
    for line in r["out_72b"].split("\n"):
        if line.strip():
            print(f"  {line}")


def print_summary(results: list[dict]):
    print(f"\n\n{SEP2}")
    print("  SUMMARY — 7B vs 72B")
    print(SEP2)
    print(f"\n  {'Case':<6} {'Purpose':<20} {'7B lat':>8} {'72B lat':>9} "
          f"{'7B nums':>9} {'72B nums':>10} {'7B terms':>10} {'72B terms':>10}")
    print(f"  {SEP}")

    tot_lat_7b = tot_lat_72b = 0
    tot_pres_7b = tot_pres_72b = tot_nums = 0
    tot_terms_7b = tot_terms_72b = 0

    for r in results:
        p7,  total = number_preservation(r["text"], r["out_7b"])
        p72, _     = number_preservation(r["text"], r["out_72b"])
        t7  = count_medical_terms(r["out_7b"])
        t72 = count_medical_terms(r["out_72b"])
        tot_lat_7b  += r["lat_7b"];  tot_lat_72b  += r["lat_72b"]
        tot_pres_7b += p7;           tot_pres_72b += p72
        tot_nums    += total
        tot_terms_7b += t7;          tot_terms_72b += t72

        print(f"  {r['id']:<6} {r['purpose']:<20} "
              f"{r['lat_7b']:>7.1f}s {r['lat_72b']:>8.1f}s "
              f"{p7}/{total:>2}{'':>6} {p72}/{total:>2}{'':>7} "
              f"{t7:>10} {t72:>10}")

    n = len(results)
    print(f"  {SEP}")
    print(f"  {'TOTAL / AVG':<26} "
          f"{tot_lat_7b/n:>7.1f}s {tot_lat_72b/n:>8.1f}s "
          f"{tot_pres_7b}/{tot_nums}{'':>3} {tot_pres_72b}/{tot_nums}{'':>4} "
          f"{tot_terms_7b:>10} {tot_terms_72b:>10}")

    print(f"""
  VERDICT
  ─────────────────────────────────────────────────────────────────
  Latency    7B avg {tot_lat_7b/n:.1f}s  vs  72B avg {tot_lat_72b/n:.1f}s
             72B is ~{tot_lat_72b/max(tot_lat_7b,0.1):.1f}x slower

  Accuracy   Numbers preserved:   7B {tot_pres_7b}/{tot_nums}  vs  72B {tot_pres_72b}/{tot_nums}
             Clinical terms used: 7B {tot_terms_7b}  vs  72B {tot_terms_72b}
""")


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    print(f"\n{SEP2}")
    print("  7B vs 72B REPHRASE BENCHMARK  —  running {n} cases in parallel pairs".format(n=len(CASES)))
    print(SEP2)
    print("  (Each case fires both models simultaneously)")

    results = []
    for case in CASES:
        print(f"\n  → Running case {case['id']}: {case['label']} ...", flush=True)
        r = await run_case(case)
        results.append(r)
        print_case(r)

    print_summary(results)

asyncio.run(main())
