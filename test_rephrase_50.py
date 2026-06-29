#!/usr/bin/env python3
"""50-case rephrase test suite — all purposes, grammar mistakes, hard/complex inputs.
Saves results to rephrase_results.csv"""

import csv
import json
import re
import time
import requests

API_URL = "http://localhost:8007/rephrase"

CASES = [
    # ── CHIEF COMPLAINT (8) ──────────────────────────────────────────────────
    dict(id=1, purpose="chief_complaint", note="heavy grammar + wrong word order",
         context={},
         text="patient he came hospital with chest pain since 3 day is very bad also left arm pain is going."),

    dict(id=2, purpose="chief_complaint", note="colloquial + breath problem described wrong",
         context={},
         text="she is having problem in breathe since yesterday night, coughing also lot, phlegm yellowish colour is coming."),

    dict(id=3, purpose="chief_complaint", note="telegraphic / fragmented",
         context={},
         text="fever. 5 days. headache. vomiting. no rash. child 8yr. not eating."),

    dict(id=4, purpose="chief_complaint", note="Hindi-English mix",
         context={},
         text="patient ko pet mein dard hai jo khana khane ke baad badh jaata hai, 2 din se hai, ulti bhi ho rahi hai."),

    dict(id=5, purpose="chief_complaint", note="run-on sentence no punctuation",
         context={},
         text="70 year old male with sudden weakness of right side of body and face drooping and speech is not coming properly since 2 hours and also he has history of BP and sugar from long time"),

    dict(id=6, purpose="chief_complaint", note="spelling errors + informal",
         context={},
         text="pt came with burining sensatin in urin since 2 dyas. frequncy increesed. no fevr. no dischrge."),

    dict(id=7, purpose="chief_complaint", note="already well-written (minimal change expected)",
         context={},
         text="Patient presents with progressive exertional dyspnea and bilateral pedal edema for the past 2 weeks."),

    dict(id=8, purpose="chief_complaint", note="double negatives + confusing structure",
         context={},
         text="patient is not having no fever but she is not feeling well and headache is not going away since not less than 4 days."),

    # ── DIAGNOSIS (7) ────────────────────────────────────────────────────────
    dict(id=9, purpose="diagnosis", note="abbreviations + spelling mistakes",
         context={"chief_complaint": "chest pain, sweating"},
         text="STEMI inferio wall suspected. pt has DM2 and HTN since 10yr. Kindey functon also afected. angioplasty done 2 yr back."),

    dict(id=10, purpose="diagnosis", note="very informal language",
         context={"chief_complaint": "cough 3 weeks, weight loss"},
         text="we think its TB. sputum test came positive 3 times. patient lost lot of weight. sugar also there."),

    dict(id=11, purpose="diagnosis", note="fragmented multi-diagnosis",
         context={},
         text="1. heart failure. 2. diabetes type 2. 3. kidney problem stage 3. 4. high BP. 5. thyroid underactive."),

    dict(id=12, purpose="diagnosis", note="wrong tense + passive voice errors",
         context={},
         text="dengue fever is being diagnosed. platelet count is fallen to 42000. NS1 antigen was came positive. patient is having warning signs."),

    dict(id=13, purpose="diagnosis", note="complex rare diagnosis colloquial",
         context={"investigations": "ANA positive, anti-dsDNA raised, complement low"},
         text="young female with butterfly rash and joint pains and hair falling and sun sensitivity. lupus we think it is."),

    dict(id=14, purpose="diagnosis", note="numeric values + abbreviations",
         context={"investigations": "FBS 280, HbA1c 11.2%"},
         text="newly diagnosed DM type 2. FBS 280 mg/dl. HbA1c 11.2%. pt is obese BMI 34. no compication yet."),

    dict(id=15, purpose="diagnosis", note="already clinical, minimal change",
         context={},
         text="Acute decompensated heart failure secondary to hypertensive heart disease. EF 30% on 2D echo."),

    # ── INVESTIGATIONS (6) ───────────────────────────────────────────────────
    dict(id=16, purpose="investigations", note="telegraphic dots + spelling",
         context={"diagnosis": "acute STEMI"},
         text="ECG done. ST elevaton II III aVF. trop I 4.8 high. CBC LFT RFT electrolites sent. 2D echo orderd. angio planed."),

    dict(id=17, purpose="investigations", note="grammar + informal",
         context={"diagnosis": "pulmonary TB"},
         text="sputum for AFB was sent 3 times and all came positive. chest xray showing cavitation in upper lobe. mantoux test also done."),

    dict(id=18, purpose="investigations", note="numbers heavy + scattered format",
         context={"diagnosis": "dengue with warning signs"},
         text="platelet 42000 (low), WBC 3200 (low), hematocrit 48% (raised), NS1 antigen positive, dengue IgM positive, liver enzymes SGOT 88 SGPT 72."),

    dict(id=19, purpose="investigations", note="abbreviation expansion needed",
         context={},
         text="CBC, LFT, RFT, TFT, lipid profile, HbA1c, urine RE/ME, USG abdomen and pelvis, 2D echo, ECG, CXR PA view ordered."),

    dict(id=20, purpose="investigations", note="wrong grammar in lab reporting",
         context={},
         text="Hb came 7.2 which is low, MCV was small at 62, serum iron was reduced, TIBC is raised, ferritin is very very low at 4."),

    dict(id=21, purpose="investigations", note="already formal",
         context={},
         text="Troponin I: 8.2 ng/mL (elevated). ECG: ST elevation in V1-V4. 2D echocardiography: anterior wall hypokinesia, EF 35%."),

    # ── MEDICATIONS (7) ──────────────────────────────────────────────────────
    dict(id=22, purpose="medications", note="wrong frequency grammar",
         context={"diagnosis": "hypertension, diabetes"},
         text="metformin 500mg two time daily, amlodipine 5mg one time morning, aspirin 75mg one time night, atorvastatin 40mg sleeping time."),

    dict(id=23, purpose="medications", note="informal dosage description",
         context={"diagnosis": "severe pneumonia"},
         text="put him on piperacillin tazobactam 4.5gm in drip every 8 hour. also azithromycin 500mg once daily tablet. oxygen also given."),

    dict(id=24, purpose="medications", note="Hindi-English medication instructions",
         context={"diagnosis": "type 2 diabetes"},
         text="metformin 500mg subah aur raat lena hai. glipizide 5mg subah khane se pahle. insulin se abhi ke liye door rahenge."),

    dict(id=25, purpose="medications", note="multiple drugs fragmented",
         context={"diagnosis": "post STEMI, on dual antiplatelet"},
         text="aspirin 75mg. clopidogrel 75mg. atorvastatin 40mg. ramipril 2.5mg. bisoprolol 2.5mg. pantoprazole 40mg. all once daily."),

    dict(id=26, purpose="medications", note="grammar mistakes in complex regimen",
         context={"diagnosis": "type 1 diabetes"},
         text="glargine 22 unit is giving at night time. aspart 6 unit is giving with each meal time. patient should monitor sugar 4 time daily."),

    dict(id=27, purpose="medications", note="wrong unit format + spelling",
         context={"diagnosis": "severe sepsis"},
         text="meropenem 1 gram intravenous every 8 hourly. vancomycin 1gm IV 12 hourly with TDM. noradrenaline 0.1 mcg/kg/min drip."),

    dict(id=28, purpose="medications", note="already formal",
         context={},
         text="Tab. Amlodipine 5mg once daily, Tab. Telmisartan 40mg once daily, Tab. Aspirin 75mg once daily after food."),

    # ── PROCEDURES (5) ───────────────────────────────────────────────────────
    dict(id=29, purpose="procedures", note="informal procedure description",
         context={"diagnosis": "acute appendicitis"},
         text="we took patient to OT and did keyhole surgery for appendix removal. everything went fine. no complication."),

    dict(id=30, purpose="procedures", note="grammar + wrong construction",
         context={"diagnosis": "left pleural effusion"},
         text="fluid was tapped from left side of chest under ultrasound guidance is done. 800ml straw colour fluid was came out and sent for test."),

    dict(id=31, purpose="procedures", note="telegraphic",
         context={"diagnosis": "STEMI"},
         text="primary PCI done. LAD stented. TIMI 3 flow achieved. cath lab time 45 min. no complication. shifted CCU."),

    dict(id=32, purpose="procedures", note="Hindi-English mix",
         context={},
         text="patient ka central line right jugular vein mein dala gaya. procedure theek se ho gayi. koi problem nahi aaya."),

    dict(id=33, purpose="procedures", note="already formal",
         context={},
         text="Diagnostic upper GI endoscopy performed under conscious sedation. Findings: duodenal ulcer 1.2cm with clean base."),

    # ── VITALS (5) ───────────────────────────────────────────────────────────
    dict(id=34, purpose="vitals", note="spelled out + wrong format",
         context={},
         text="BP 160 by 100 mmhg. pulse 98 per minute and it is irregular. oxygen 93 percentage on room air. temprature 100.4 fehrenheit. RR 22 per min."),

    dict(id=35, purpose="vitals", note="mixed units + grammar",
         context={},
         text="BP is 158/96 mmHg. heart rate 104 per min. spo2 94% on oxygen 4 litre. temp is 101 degree F. weight 87 kilogram."),

    dict(id=36, purpose="vitals", note="scattered no structure",
         context={},
         text="random sugar 312. height 165cm. weight 87kg. BMI 32. BP 142/90. pulse 88. rr 18. temp 98.6F. spo2 99%."),

    dict(id=37, purpose="vitals", note="child vitals + informal",
         context={},
         text="child age 5yr. weight 18kg. temp 39.9 degree celsius. heart rate 136/min. rr 42/min. spo2 87% room air. bp 90/60."),

    dict(id=38, purpose="vitals", note="already formal",
         context={},
         text="BP 118/76 mmHg, HR 72/min regular, SpO2 98% on room air, Temperature 37.1°C, RR 16/min, Weight 68 kg."),

    # ── ADVICE & FOLLOW-UP (5) ───────────────────────────────────────────────
    dict(id=39, purpose="advice_followup", note="Hindi-English mix",
         context={"diagnosis": "diabetes, hypertension"},
         text="patient ko bola ki dawai regularly lena. BP aur sugar ghar pe check karte rehna. namak aur cheeni kam karo. roz 30 min walk karo. ek hafte baad aana."),

    dict(id=40, purpose="advice_followup", note="grammar mistakes",
         context={"diagnosis": "post appendectomy"},
         text="patient should not do heavy work for 2 week. wound should kept dry. daily dressing is done. come back if fever is come or wound is becoming red."),

    dict(id=41, purpose="advice_followup", note="telegraphic bullet style",
         context={"diagnosis": "pulmonary TB"},
         text="DOTS therapy 6 months. no alcohol. regular follow up monthly. check liver function 1 month. report side effects. contact tracing family members."),

    dict(id=42, purpose="advice_followup", note="informal patient education",
         context={"diagnosis": "dengue"},
         text="drink lot of water and ORS. rest at home. do not take aspirin or brufen for fever. take paracetamol only. come to hospital immediately if bleeding or belly pain very bad."),

    dict(id=43, purpose="advice_followup", note="already formal",
         context={},
         text="Patient advised dietary modification: low salt, low fat diet. Exercise 30 minutes daily. Follow-up in 2 weeks with BP monitoring diary."),

    # ── DOCTOR'S NOTES (7) ───────────────────────────────────────────────────
    dict(id=44, purpose="doctors_notes", note="very informal + grammar disaster (→ 72B)",
         context={"chief_complaint": "chest pain", "diagnosis": "STEMI"},
         text="65 yr old fat male came in with chest hurting real bad since 2 days sweating a lot feels like something heavy on chest his BP was 162/98 pulse running at 112/min O2 sat dropped to 91% fever 101.4F we gave him aspirin 325mg right away and put him on oxygen 4L/min thru nasal prongs ECG showed some ST going up in leads 2,3,avF troponin came back high at 2.8 told family looks like heart attack"),

    dict(id=45, purpose="doctors_notes", note="fragmented + telegraphic (→ 72B)",
         context={},
         text="52F. HTN. DM2. came with breathlessness 2 days. pedal edema bilateral. JVP raised. basal crepts. CXR cardiomegaly. BNP 1240. echo EF 28%. started on furosemide 40mg IV. fluid restricted 1L/day. cardiology referral done."),

    dict(id=46, purpose="doctors_notes", note="wrong word order + subject errors (→ 72B)",
         context={},
         text="patient whose age 45 year male is having jaundice which is yellowish colour of eyes and skin since 1 week. dark urine is passing. pale stool is coming. liver function test total bilirubin 14.2 came. hepatitis B surface antigen positive came. ultrasound showing liver enlarged with heterogeneous echotexture."),

    dict(id=47, purpose="doctors_notes", note="short note — grammar only (→ 7B, under 50 words)",
         context={},
         text="patient is stable now. pain is controlled. vitals is normal. plan is to discharge tomorrow if no complication."),

    dict(id=48, purpose="doctors_notes", note="ICU note with numbers (→ 72B)",
         context={},
         text="pt on ventilator day 3. PEEP 8, FiO2 50%, TV 480ml. ABG pH 7.38 pCO2 42 pO2 88. sedation fentanyl 25mcg/hr midazolam 2mg/hr. vasopressor noradrenaline 0.08 mcg/kg/min. creatinine rising 1.8 today from 1.2 yesterday. urine output 20ml/hr. nephrology consulted."),

    dict(id=49, purpose="doctors_notes", note="already well-written (→ 7B, under 50 words)",
         context={},
         text="Patient is hemodynamically stable. Pain adequately controlled with analgesics. Wound is clean and dry. Tolerating oral feeds. Planned for discharge tomorrow with outpatient follow-up in 1 week."),

    # ── EDGE / STRESS CASES (1) ───────────────────────────────────────────────
    dict(id=50, purpose="chief_complaint", note="GARBAGE INPUT — should return fallback",
         context={},
         text="asdfgh qwerty zxcvbn mnbvcd poiuyt lkjhg"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def number_preservation(original: str, output: str):
    orig = {n for n in re.findall(r"\d+(?:\.\d+)?", original) if float(n) >= 10 or "." in n}
    out  = set(re.findall(r"\d+(?:\.\d+)?", output))
    preserved = len(orig & out)
    return preserved, len(orig)


_DOSAGE_RE = re.compile(
    r"\b([A-Za-z][a-z]{2,})\s+\d+\s*(?:mg|mcg|g\b|ml|mL|L\b|IU|units?|tabs?|caps?)",
    re.IGNORECASE,
)

def drug_preservation(original: str, output: str) -> str:
    drugs = {m.lower() for m in _DOSAGE_RE.findall(original)}
    if not drugs:
        return "N/A"
    out_lower = output.lower()
    missing = [d for d in drugs if d not in out_lower]
    return "PASS" if not missing else f"MISSING: {missing}"


# ── Run one case ──────────────────────────────────────────────────────────────
def run_case(case: dict) -> dict:
    payload = {
        "text": case["text"],
        "purpose": case["purpose"],
        "context": case.get("context", {}),
        "patient_id": f"test-{case['id']}",
    }
    start = time.monotonic()
    resp = None
    # Retry up to 6 times on 429 — each wait clears the 10/min window
    for attempt in range(6):
        try:
            resp = requests.post(API_URL, json=payload, timeout=90)
            if resp.status_code == 429:
                wait = 12 * (attempt + 1)
                print(f"           ⏳ rate limited, waiting {wait}s ...", flush=True)
                time.sleep(wait)
                continue
            break
        except Exception as e:
            if attempt == 5:
                duration_ms = round((time.monotonic() - start) * 1000)
                return {
                    "id": case["id"], "purpose": case["purpose"],
                    "note": case.get("note", ""),
                    "input_words": len(case["text"].split()),
                    "input_text": case["text"][:120],
                    "rephrased": "", "fallback": False,
                    "duration_ms": duration_ms,
                    "nums_preserved": "N/A", "drug_check": "N/A",
                    "status": f"ERROR: {e}",
                }
            time.sleep(12)

    duration_ms = round((time.monotonic() - start) * 1000)
    if resp is None:
        return {
            "id": case["id"], "purpose": case["purpose"],
            "note": case.get("note", ""),
            "input_words": len(case["text"].split()),
            "input_text": case["text"][:120],
            "rephrased": "", "fallback": False,
            "duration_ms": duration_ms,
            "nums_preserved": "N/A", "drug_check": "N/A",
            "status": "NO_RESPONSE",
        }

    if resp.status_code == 200:
        data = resp.json()
        rephrased  = data.get("rephrased", "")
        fallback   = data.get("fallback", False)
        nums_ok, nums_total = number_preservation(case["text"], rephrased)
        drug_ok    = drug_preservation(case["text"], rephrased)
        status     = "FALLBACK" if fallback else "PASS"
        return {
            "id": case["id"],
            "purpose": case["purpose"],
            "note": case.get("note", ""),
            "input_words": len(case["text"].split()),
            "input_text": case["text"][:120],
            "rephrased": rephrased[:200],
            "fallback": fallback,
            "duration_ms": duration_ms,
            "nums_preserved": f"{nums_ok}/{nums_total}",
            "drug_check": drug_ok,
            "status": status,
        }
    else:
        return {
            "id": case["id"], "purpose": case["purpose"],
            "note": case.get("note", ""),
            "input_words": len(case["text"].split()),
            "input_text": case["text"][:120],
            "rephrased": "", "fallback": False,
            "duration_ms": duration_ms,
            "nums_preserved": "N/A", "drug_check": "N/A",
            "status": f"HTTP {resp.status_code}",
        }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\nRunning {len(CASES)} rephrase test cases against {API_URL}\n")
    results = []
    counts  = {"PASS": 0, "FALLBACK": 0, "ERROR": 0}
    total_ms = 0

    for i, case in enumerate(CASES, 1):
        print(f"  [{i:02d}/{len(CASES)}] {case['purpose']:<20} | {case.get('note','')[:50]}")
        r = run_case(case)
        results.append(r)

        s = r["status"]
        if s == "PASS":       counts["PASS"]     += 1
        elif s == "FALLBACK": counts["FALLBACK"]  += 1
        else:                 counts["ERROR"]     += 1
        total_ms += r["duration_ms"]

        print(f"           → [{s}] {r['duration_ms']}ms | "
              f"nums {r['nums_preserved']} | drugs {r['drug_check']}")
        print(f"              {r['rephrased'][:90]!r}")

    # Save CSV
    csv_path = "rephrase_results.csv"
    fields = ["id","purpose","note","input_words","input_text","rephrased",
              "fallback","duration_ms","nums_preserved","drug_check","status"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    avg_ms = total_ms // len(CASES)
    pass_  = counts["PASS"]
    fb_    = counts["FALLBACK"]
    err_   = counts["ERROR"]

    print(f"\n{'='*65}")
    print(f"  Total cases : {len(CASES)}")
    print(f"  PASS        : {pass_}")
    print(f"  FALLBACK    : {fb_}  (garbage correctly rejected)")
    print(f"  ERROR       : {err_}")
    print(f"  Avg latency : {avg_ms} ms")
    print(f"  CSV saved   : {csv_path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
