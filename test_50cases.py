#!/usr/bin/env python3
"""50-case test suite for AutoSuggestion API. Saves results to test_results.csv"""

import csv
import json
import time
import requests

API_URL = "http://localhost:8008/suggest"

CASES = [
    # ── chief_complaint (10) ─────────────────────────────────────────────────
    dict(id=1, purpose="chief_complaint",
         prefix="patient comes in complaining of",
         full_note="45 year old male visited OPD. bilateral leg swelling noted for past 2 weeks. no fever.",
         context={"chief_complaint": "bilateral leg swelling"}),

    dict(id=2, purpose="chief_complaint",
         prefix="she presents with severe",
         full_note="28 year old female with headache since 3 days. associated nausea and vomiting. photophobia present.",
         context={"chief_complaint": "severe headache with vomiting and photophobia"}),

    dict(id=3, purpose="chief_complaint",
         prefix="patient reports crushing chest",
         full_note="60 year old male with chest pain radiating to left arm. diaphoresis present. onset 2 hours ago.",
         context={"chief_complaint": "chest pain radiating to left arm", "diagnosis": "suspected STEMI"}),

    dict(id=4, purpose="chief_complaint",
         prefix="child was brought in with",
         full_note="5 year old child with high fever since 4 days. rash appeared on day 3. reduced appetite and activity.",
         context={"chief_complaint": "high fever with skin rash"}),

    dict(id=5, purpose="chief_complaint",
         prefix="patient has been having burning",
         full_note="32 year old female with burning micturition since 2 days. increased frequency. cloudy urine. no fever.",
         context={"chief_complaint": "burning urination with frequency"}),

    dict(id=6, purpose="chief_complaint",
         prefix="elderly patient brought with sudden",
         full_note="70 year old hypertensive female with sudden onset right-sided weakness. slurred speech. onset 3 hours ago.",
         context={"chief_complaint": "sudden right-sided weakness and slurred speech"}),

    dict(id=7, purpose="chief_complaint",
         prefix="she complains of colicky",
         full_note="40 year old female with intermittent colicky abdominal pain. worse after fatty meals. nausea and vomiting.",
         context={"chief_complaint": "colicky pain after fatty food"}),

    dict(id=8, purpose="chief_complaint",
         prefix="patient has had productive cough",
         full_note="25 year old male with productive cough for 3 weeks. low grade fever. night sweats. significant weight loss.",
         context={"chief_complaint": "cough with night sweats and weight loss"}),

    dict(id=9, purpose="chief_complaint",
         prefix="he is experiencing progressive",
         full_note="55 year old diabetic male with progressive breathlessness on exertion. orthopnea noted. bilateral pedal edema present.",
         context={"chief_complaint": "breathlessness on exertion", "diagnosis": "heart failure"}),

    dict(id=10, purpose="chief_complaint",
         prefix="patient noticed painless",
         full_note="50 year old male with painless swelling in neck for 3 months. gradually increasing in size. no pain or fever.",
         context={"chief_complaint": "painless neck swelling growing slowly"}),

    # ── diagnosis (7) ────────────────────────────────────────────────────────
    dict(id=11, purpose="diagnosis",
         prefix="provisional diagnosis is",
         full_note="patient with productive cough 3 weeks, low grade fever, night sweats, weight loss. sputum AFB positive.",
         context={"diagnosis": "pulmonary tuberculosis", "investigations": "sputum AFB positive x3"}),

    dict(id=12, purpose="diagnosis",
         prefix="patient is newly diagnosed with",
         full_note="65 year old male with polyuria, polydipsia, weight loss of 8kg in 2 months. FBS 280 mg/dl. HbA1c 11.2%.",
         context={"diagnosis": "type 2 diabetes mellitus", "investigations": "FBS 280, HbA1c 11.2%"}),

    dict(id=13, purpose="diagnosis",
         prefix="clinical picture is consistent with acute",
         full_note="18 year old male with RLQ pain, rebound tenderness, fever 38.5C. WBC 15000. rovsing sign positive.",
         context={"diagnosis": "acute appendicitis"}),

    dict(id=14, purpose="diagnosis",
         prefix="ECG and troponin findings confirm",
         full_note="60 year old male with crushing chest pain 2 hours duration. ST elevation in V1-V4. troponin I elevated at 8.2.",
         context={"diagnosis": "acute anterior STEMI", "investigations": "ECG ST elevation V1-V4, troponin 8.2"}),

    dict(id=15, purpose="diagnosis",
         prefix="respiratory examination suggests",
         full_note="fever, productive cough with rusty sputum, decreased breath sounds right lower zone, dullness on percussion.",
         context={"diagnosis": "right lower lobe pneumonia", "investigations": "CXR consolidation right lower zone"}),

    dict(id=16, purpose="diagnosis",
         prefix="the patient has been diagnosed with",
         full_note="young female with butterfly rash on face, joint pain, photosensitivity, hair loss. ANA positive. anti-dsDNA elevated.",
         context={"diagnosis": "SLE", "investigations": "ANA positive 1:320, anti-dsDNA positive"}),

    dict(id=17, purpose="diagnosis",
         prefix="based on findings diagnosis is",
         full_note="high fever 5 days, thrombocytopenia platelet 42000, positive tourniquet test, NS1 antigen positive.",
         context={"diagnosis": "dengue fever with warning signs", "investigations": "platelet 42000, NS1 positive"}),

    # ── investigations (6) ───────────────────────────────────────────────────
    dict(id=18, purpose="investigations",
         prefix="we need to urgently send",
         full_note="chest pain with shortness of breath. tachycardia HR 115. SpO2 91%. suspect pulmonary embolism.",
         context={"investigations": "CT pulmonary angiography, D-dimer, ABG"}),

    dict(id=19, purpose="investigations",
         prefix="liver function tests along with",
         full_note="fever with jaundice since 5 days. hepatomegaly. dark urine. pale stools. bilirubin elevated.",
         context={"investigations": "LFT, hepatitis B and C serology, USG abdomen"}),

    dict(id=20, purpose="investigations",
         prefix="ultrasound abdomen shows",
         full_note="RUQ pain after meals, nausea, fatty food intolerance. murphy sign positive. fever 37.9C.",
         context={"investigations": "USG abdomen", "diagnosis": "acute cholecystitis"}),

    dict(id=21, purpose="investigations",
         prefix="blood culture and sensitivity for",
         full_note="high grade fever with rigors. sweating. spleen enlarged. peripheral smear sent. malaria suspected.",
         context={"investigations": "peripheral blood smear, malaria antigen, blood culture"}),

    dict(id=22, purpose="investigations",
         prefix="CBC report reveals",
         full_note="fatigue, pallor, koilonychia, spoon-shaped nails. dietary history poor. suspected iron deficiency.",
         context={"investigations": "CBC, serum ferritin, TIBC", "diagnosis": "iron deficiency anemia"}),

    dict(id=23, purpose="investigations",
         prefix="MRI brain ordered to rule out",
         full_note="sudden onset severe headache described as thunderclap. worst headache of life. neck stiffness. CT normal.",
         context={"investigations": "MRI brain with angiography, lumbar puncture"}),

    # ── medications (7) ──────────────────────────────────────────────────────
    dict(id=24, purpose="medications",
         prefix="start patient on tablet amlodipine",
         full_note="hypertensive patient BP 168/104 on 3 readings. no previous antihypertensives. no contraindications.",
         context={"medications": "amlodipine 5mg once daily", "diagnosis": "essential hypertension"}),

    dict(id=25, purpose="medications",
         prefix="patient should take metformin",
         full_note="newly diagnosed type 2 diabetic. BMI 32. no renal impairment. eGFR 85. FBS 245 mg/dl.",
         context={"medications": "metformin 500mg twice daily", "diagnosis": "type 2 DM"}),

    dict(id=26, purpose="medications",
         prefix="IV piperacillin tazobactam started for",
         full_note="severe community acquired pneumonia. febrile 39.5C. SpO2 88% on room air. CXR bilateral infiltrates.",
         context={"medications": "piperacillin-tazobactam 4.5g IV every 8 hours", "diagnosis": "severe CAP"}),

    dict(id=27, purpose="medications",
         prefix="add omeprazole to protect against",
         full_note="patient on diclofenac for osteoarthritis. now has epigastric pain and heartburn. h.pylori test ordered.",
         context={"medications": "omeprazole 20mg before breakfast", "diagnosis": "NSAID gastropathy"}),

    dict(id=28, purpose="medications",
         prefix="dual antiplatelet therapy with",
         full_note="post STEMI day 1. primary PCI done. stent placed in LAD. no bleeding risk identified.",
         context={"medications": "aspirin 75mg + clopidogrel 75mg daily", "diagnosis": "post STEMI"}),

    dict(id=29, purpose="medications",
         prefix="insulin regimen adjusted to",
         full_note="type 1 diabetic. CBG 340 fasting. HbA1c 10.8%. currently on glargine 18 units. poor control.",
         context={"medications": "glargine 22 units night, aspart 6 units with meals", "diagnosis": "type 1 DM"}),

    dict(id=30, purpose="medications",
         prefix="antibiotic for UTI is",
         full_note="UTI confirmed. urine culture grew E.coli. sensitive to nitrofurantoin and trimethoprim. resistant to amoxicillin.",
         context={"medications": "nitrofurantoin 100mg twice daily for 5 days", "diagnosis": "uncomplicated UTI"}),

    # ── procedures (5) ───────────────────────────────────────────────────────
    dict(id=31, purpose="procedures",
         prefix="patient underwent emergency laparoscopic",
         full_note="acute appendicitis confirmed. taken to OR on emergency basis. general anesthesia administered. no perforation.",
         context={"procedures": "laparoscopic appendectomy"}),

    dict(id=32, purpose="procedures",
         prefix="central venous access secured via",
         full_note="septic patient in ICU. peripheral veins collapsed. CVP monitoring needed. antibiotics and vasopressors running.",
         context={"procedures": "right internal jugular central line insertion"}),

    dict(id=33, purpose="procedures",
         prefix="thoracocentesis performed and",
         full_note="left pleural effusion confirmed on CXR and USG. dyspnea at rest. SpO2 91%. diagnostic tap planned.",
         context={"procedures": "thoracocentesis", "investigations": "pleural fluid sent for analysis"}),

    dict(id=34, purpose="procedures",
         prefix="wound debridement done under",
         full_note="diabetic foot ulcer grade 3. necrotic tissue present. bone not exposed. wound culture sent.",
         context={"procedures": "wound debridement under local anesthesia"}),

    dict(id=35, purpose="procedures",
         prefix="nasogastric tube inserted for",
         full_note="acute pancreatitis. nil by mouth. severe vomiting. abdominal distension. lipase 1200.",
         context={"procedures": "NGT insertion for gastric decompression"}),

    # ── vitals (5) ───────────────────────────────────────────────────────────
    dict(id=36, purpose="vitals",
         prefix="blood pressure is recorded as",
         full_note="hypertensive urgency. severe headache. no focal neurological deficits. on antihypertensives.",
         context={"vitals": "BP 192/112 mmHg, HR 92, SpO2 98%"}),

    dict(id=37, purpose="vitals",
         prefix="temperature is elevated at",
         full_note="febrile patient with chills and rigors. malaria suspected. blood smear sent. sweating profusely.",
         context={"vitals": "temp 39.9C, HR 110, BP 108/70, RR 22"}),

    dict(id=38, purpose="vitals",
         prefix="oxygen saturation on room air",
         full_note="COPD exacerbation. increased breathlessness since morning. wheeze bilateral. on salbutamol nebulization.",
         context={"vitals": "SpO2 82%, RR 30, HR 108, temp 37.5C"}),

    dict(id=39, purpose="vitals",
         prefix="pulse rate is irregular at",
         full_note="patient with palpitations since 2 days. ECG shows irregular rhythm. atrial fibrillation suspected. no chest pain.",
         context={"vitals": "HR 134 irregular, BP 98/68, SpO2 96%"}),

    dict(id=40, purpose="vitals",
         prefix="respiratory rate is elevated",
         full_note="severe pneumonia in child. tachypnea with grunting. subcostal retractions. SpO2 falling. ICU transfer arranged.",
         context={"vitals": "RR 42/min, SpO2 87%, temp 39.3C, HR 136"}),

    # ── advice_followup (6) ──────────────────────────────────────────────────
    dict(id=41, purpose="advice_followup",
         prefix="patient must complete full course of",
         full_note="TB patient started on DOTS regimen. counselled on treatment compliance and side effects. no alcohol.",
         context={"advice_followup": "complete 6 months DOTS therapy without break"}),

    dict(id=42, purpose="advice_followup",
         prefix="follow up appointment scheduled after",
         full_note="post appendectomy day 2. wound healthy. diet tolerated. vital signs stable. sutures intact.",
         context={"advice_followup": "follow up after 7 days for suture removal"}),

    dict(id=43, purpose="advice_followup",
         prefix="dietary advice includes avoiding",
         full_note="newly diagnosed type 2 diabetic. obese BMI 33. counselled on lifestyle modification.",
         context={"advice_followup": "avoid sugar, refined carbs, processed food. increase fiber and vegetables."}),

    dict(id=44, purpose="advice_followup",
         prefix="patient should return to emergency if",
         full_note="dengue with warning signs discharged. red flag signs explained. strict intake-output monitoring at home.",
         context={"advice_followup": "return immediately for severe abdominal pain, bleeding, or difficulty breathing"}),

    dict(id=45, purpose="advice_followup",
         prefix="post MI counselling includes",
         full_note="post MI discharge. PCI done. dual antiplatelet started. cardiac rehab discussed. smoker advised to quit.",
         context={"advice_followup": "smoking cessation, cardiac rehab, medication compliance, avoid heavy lifting"}),

    dict(id=46, purpose="advice_followup",
         prefix="wound care at home should include",
         full_note="surgical wound clean. discharged day 3. no signs of infection. dressing changed daily in hospital.",
         context={"advice_followup": "daily dressing change, keep wound dry, no swimming, return if redness or discharge"}),

    # ── doctors_notes (5) ────────────────────────────────────────────────────
    dict(id=47, purpose="doctors_notes",
         prefix="clinically patient appears to be",
         full_note="78 year old with multiple comorbidities. on 8 medications. poor functional status. lives alone.",
         context={"doctors_notes": "frail elderly, high fall risk, palliative approach discussed with family"}),

    dict(id=48, purpose="doctors_notes",
         prefix="in view of deteriorating condition",
         full_note="ICU patient on ventilator day 6. no neurological improvement. family meeting held. prognosis poor.",
         context={"doctors_notes": "guarded prognosis discussed, family counselled, DNR order considered"}),

    dict(id=49, purpose="doctors_notes",
         prefix="referral made to cardiology for",
         full_note="patient with refractory heart failure despite optimal medical therapy. EF 25%. NYHA class IV.",
         context={"doctors_notes": "referred to cardiology for advanced heart failure management and device therapy"}),

    # ── edge / stress cases (3) ──────────────────────────────────────────────
    dict(id=50, purpose="chief_complaint",
         prefix="abc xyz def",
         full_note="gibberish patient note with no clinical meaning. random words here.",
         context={"chief_complaint": "nonsense data test"},
         note="GARBAGE INPUT"),

    dict(id=51, purpose="medications",
         prefix="patient needs medication for",
         full_note="patient has fever. no other details available.",
         context={},
         note="EMPTY CONTEXT"),

    dict(id=52, purpose="diagnosis",
         prefix="diagnosis is conflicting as",
         full_note="patient has both high and low BP readings. contradictory findings. note written in error.",
         context={"diagnosis": "hypertension", "chief_complaint": "hypotension and dizziness"},
         note="CONFLICTING DATA"),
]


def run_case(case: dict) -> dict:
    payload = {
        "prefix": case["prefix"],
        "full_note": case["full_note"],
        "patient_id": f"test-{case['id']}",
        "stage_id": case["id"],
        "purpose": case["purpose"],
        "debug": False,
        "context": case.get("context", {}),
    }
    start = time.monotonic()
    try:
        resp = requests.post(API_URL, json=payload, timeout=30)
        duration_ms = round((time.monotonic() - start) * 1000)
        if resp.status_code == 200:
            data = resp.json()
            suggestion = data.get("suggestion", "")
            status = "EMPTY" if suggestion == "" else "PASS"
            return {
                "id": case["id"],
                "purpose": case["purpose"],
                "prefix": case["prefix"],
                "full_note": case["full_note"][:80] + "..." if len(case["full_note"]) > 80 else case["full_note"],
                "context": json.dumps(case.get("context", {})),
                "suggestion": suggestion,
                "full_text": data.get("full_text", ""),
                "duration_ms": duration_ms,
                "status": status,
                "note": case.get("note", ""),
            }
        else:
            return {
                "id": case["id"],
                "purpose": case["purpose"],
                "prefix": case["prefix"],
                "full_note": case["full_note"][:80],
                "context": json.dumps(case.get("context", {})),
                "suggestion": "",
                "full_text": "",
                "duration_ms": duration_ms,
                "status": f"ERROR {resp.status_code}",
                "note": case.get("note", ""),
            }
    except Exception as e:
        duration_ms = round((time.monotonic() - start) * 1000)
        return {
            "id": case["id"],
            "purpose": case["purpose"],
            "prefix": case["prefix"],
            "full_note": case["full_note"][:80],
            "context": json.dumps(case.get("context", {})),
            "suggestion": "",
            "full_text": "",
            "duration_ms": duration_ms,
            "status": f"EXCEPTION: {e}",
            "note": case.get("note", ""),
        }


def main():
    print(f"Running {len(CASES)} test cases against {API_URL}\n")
    results = []
    counts = {"PASS": 0, "EMPTY": 0, "ERROR": 0}
    total_ms = 0

    for i, case in enumerate(CASES, 1):
        tag = f"[{case.get('note', '')}]" if case.get("note") else ""
        print(f"  [{i:02d}/{len(CASES)}] {case['purpose']:<20} prefix: {case['prefix'][:40]!r} {tag}")
        result = run_case(case)
        results.append(result)

        status = result["status"]
        if status == "PASS":
            counts["PASS"] += 1
        elif status == "EMPTY":
            counts["EMPTY"] += 1
        else:
            counts["ERROR"] += 1

        total_ms += result["duration_ms"]
        print(f"         → [{status}] {result['duration_ms']}ms | {result['suggestion'][:70]!r}")

    # Save CSV
    csv_path = "test_results.csv"
    fieldnames = ["id", "purpose", "prefix", "full_note", "context", "suggestion", "full_text", "duration_ms", "status", "note"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*60}")
    print(f"  Total : {len(CASES)}")
    print(f"  PASS  : {counts['PASS']}")
    print(f"  EMPTY : {counts['EMPTY']}")
    print(f"  ERROR : {counts['ERROR']}")
    print(f"  Avg latency : {total_ms // len(CASES)} ms")
    print(f"  Results saved → {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
