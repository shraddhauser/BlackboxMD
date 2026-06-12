#  BLACKBOX MD
### Explainable AI Second Opinion System for Misdiagnosis Prevention

> *"The doctor isn't wrong. The AI just wants to show its work."*

---

##  The Problem

**12 million diagnostic errors** occur every year in the US alone. Globally, misdiagnosis is the leading cause of preventable patient harm — more than surgical errors, medication errors, or hospital-acquired infections combined.

Existing clinical AI tools (Epic's sepsis model, IBM Watson Health) were shut down or abandoned — not because the AI was wrong, but because **doctors couldn't trust what they couldn't explain.** A black box saying "this patient has cancer" with no reasoning is worse than useless in a clinical setting. Doctors ignored it. Patients died.

**Watson Health was shut down in 2022 for exactly this reason.**

The problem was never the AI. It was the silence inside it.

---

##  The Solution

BLACKBOX MD is an **explainable AI second opinion layer** that sits between the doctor's EHR (Electronic Health Record) system and the patient file.

It does not replace the doctor.
It does not override any decision.
It **raises a quiet flag** — and shows exactly why.

When a doctor's diagnosis statistically conflicts with the patient's symptom pattern, lab trends, and 10 million similar historical cases, BLACKBOX MD surfaces a ranked list of alternative diagnoses with a plain-language explanation of every data point that triggered the flag.

```
    Flag raised for Patient #4821
    Primary diagnosis:  Tension Headache
    Flagged because:    3 data points conflict with this diagnosis

    → Serum sodium: 128 mEq/L  (low — consistent with hyponatremia)
    → Symptom onset: gradual over 6 days  (atypical for tension headache)
    → Age + gender profile: matches 74% of missed SAH cases in dataset

    Suggested review:   Subarachnoid Hemorrhage (SAH) — confidence: 67%
                        Hyponatremia-induced headache — confidence: 52%
```

The doctor sees the reasoning. The doctor decides. BLACKBOX MD just makes sure nothing was silently missed.

---

##  Why This Doesn't Exist Yet

| Barrier | Why it blocked everyone before | How BLACKBOX MD addresses it |
|---|---|---|
| Black-box AI | Doctors don't trust output they can't verify | XAI layer (SHAP/LIME) surfaces feature-level reasoning in plain English |
| EHR integration | Epic, Cerner APIs are proprietary and hostile | HL7 FHIR R4 standard — works with any FHIR-compliant EHR |
| Legal liability | If AI is wrong, who is responsible? | System is advisory only — flags never override; doctor retains full authority |
| Data availability | No annotated misdiagnosis dataset exists | Trained on MIMIC-IV (50K+ ICU records) + retrospective diagnosis correction data |
| Watson Health | Tried to replace doctors, not assist them | Designed as a background flag, not a foreground decision-maker |

---

##  System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        EHR SYSTEM                           │
│              (Epic / Cerner / any FHIR R4)                  │
└────────────────────────┬────────────────────────────────────┘
                         │  FHIR R4 API (HL7 standard)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   BLACKBOX MD ENGINE                        │
│                                                             │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐  │
│  │ Data Ingestion│→│  ML Classifier│→│  XAI Explainer  │  │
│  │  (FHIR R4)  │  │ (XGBoost +   │  │ (SHAP values +  │  │
│  │             │  │  symptom NLP) │  │  plain language │  │
│  └─────────────┘  └──────────────┘  │  generator)     │  │
│                                     └────────┬────────┘  │
│                                              │             │
│  ┌────────────────────────────────────────── ▼──────────┐  │
│  │              ALERT LAYER                             │  │
│  │  Severity scoring → EHR notification → audit log    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                         │  Advisory flag only
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     DOCTOR DASHBOARD                        │
│         Flags · Explanations · Suggested alternatives       │
└─────────────────────────────────────────────────────────────┘
```

---

##  Core Technical Components

### 1. Diagnosis Classifier
- Model: **XGBoost** ensemble + symptom NLP pipeline (BioBERT fine-tuned on clinical notes)
- Input features: vitals, lab values, symptom duration, patient demographics, medication history, prior diagnoses
- Training data: **MIMIC-IV** (PhysioNet) — 50,000+ ICU patient records with retrospective diagnosis corrections
- Output: probability distribution over 1,200+ ICD-10 diagnosis codes

### 2. Explainability Layer (the "un-blackboxing")
- **SHAP (SHapley Additive exPlanations)** — assigns contribution score to every input feature
- **LIME (Local Interpretable Model-Agnostic Explanations)** — local approximation for individual predictions
- Plain-language generator converts SHAP values into human-readable clinical sentences
- Every flag includes: top 3 contributing features, confidence score, count of similar cases in training data

### 3. EHR Integration
- Protocol: **HL7 FHIR R4** (universal healthcare interoperability standard)
- Supported resources: `Patient`, `Observation`, `Condition`, `DiagnosticReport`, `MedicationRequest`
- Real-time webhook listener on EHR diagnosis submission events
- Async — never blocks the doctor's workflow

### 4. Alert Interface
- Embedded panel in existing EHR UI (no new app for doctors to learn)
- Severity levels: `REVIEW SUGGESTED` / `FLAG` / `URGENT FLAG`
- All flags are logged to immutable audit trail (HIPAA compliance)
- Doctor can dismiss with one click — no friction

---

## Target Impact

| Metric | Current State | With BLACKBOX MD |
|---|---|---|
| Diagnostic errors (US/yr) | 12 million | Est. 30–40% reduction in flagged categories |
| Avg. time to correct misdiagnosis | 3–5 days | Same visit (real-time flag) |
| Doctor trust in AI recommendations | Low (Watson failure) | High — explanation visible, authority retained |
| EHR integration time | Months (custom) | Days (FHIR R4 standard) |

---

##  Project Structure

```
blackbox-md/
├── ingestion/
│   ├── fhir_client.py          # FHIR R4 data fetcher
│   ├── patient_builder.py      # Assembles feature vector from FHIR resources
│   └── lab_normalizer.py       # Normalizes lab values across units
│
├── model/
│   ├── classifier.py           # XGBoost + BioBERT ensemble
│   ├── train.py                # Training pipeline (MIMIC-IV)
│   ├── evaluate.py             # Precision/recall by ICD-10 category
│   └── saved_models/           # Pre-trained weights
│
├── explainer/
│   ├── shap_explainer.py       # SHAP value computation
│   ├── lime_explainer.py       # LIME local approximation
│   └── language_generator.py   # SHAP → plain English sentences
│
├── alerts/
│   ├── severity_scorer.py      # Converts confidence delta to severity level
│   ├── ehr_notifier.py         # Pushes flag back to EHR via FHIR
│   └── audit_logger.py         # Immutable flag audit trail
│
├── api/
│   ├── main.py                 # FastAPI server
│   ├── routes/
│   │   ├── analyze.py          # POST /analyze — submit patient record
│   │   └── flags.py            # GET /flags/{patient_id} — retrieve flags
│   └── middleware/
│       └── hipaa_filter.py     # PHI stripping for logs
│
├── dashboard/                  # React frontend (EHR embed panel)
│   ├── FlagCard.jsx
│   ├── ExplanationPanel.jsx
│   └── AuditLog.jsx
│
├── data/
│   └── mimic_preprocessor.py   # MIMIC-IV → training format converter
│
├── tests/
│   ├── test_classifier.py
│   ├── test_explainer.py
│   └── test_fhir_integration.py
│
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

##  Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+ (dashboard only)
- Docker + Docker Compose
- Access to a FHIR R4 sandbox (recommended: [HAPI FHIR public server](https://hapi.fhir.org/))

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/blackbox-md.git
cd blackbox-md

# Install Python dependencies
pip install -r requirements.txt

# Start all services
docker-compose up --build
```

### Run with HAPI FHIR sandbox (no EHR needed)

```bash
# Load sample patient data into HAPI FHIR
python data/load_sample_patients.py --server https://hapi.fhir.org/baseR4

# Run the analysis engine
python api/main.py

# Submit a patient for analysis
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "demo-001", "fhir_server": "https://hapi.fhir.org/baseR4"}'
```

### Train on MIMIC-IV

```bash
# Download MIMIC-IV from PhysioNet (requires credentialed access)
# https://physionet.org/content/mimiciv/

python data/mimic_preprocessor.py --input ./mimic-iv/ --output ./data/processed/
python model/train.py --data ./data/processed/ --output ./model/saved_models/
```

---

## Current Status

| Component | Status |
|---|---|
| FHIR R4 data ingestion | ✅ Complete |
| XGBoost classifier (MIMIC-IV) | ✅ Complete |
| SHAP explainability layer | ✅ Complete |
| Plain-language generator | 🔄 In progress |
| EHR alert push (FHIR write) | 🔄 In progress |
| React dashboard embed | 📋 Planned |
| Clinical validation study | 📋 Planned |

---

##  Ethics & Compliance

- **Advisory only** — BLACKBOX MD never blocks, overrides, or delays any clinical action. Every flag is a suggestion.
- **HIPAA-aware design** — PHI is never logged; audit trail stores only flag metadata and anonymized feature weights.
- **Bias monitoring** — model performance is evaluated separately across age, gender, and ethnicity groups in MIMIC-IV to detect and flag disparate error rates.
- **Doctor authority is absolute** — if a doctor dismisses a flag, that dismissal is recorded and respected. The system does not re-flag the same case.

---

## 📚 References & Datasets

- **MIMIC-IV** — Johnson et al., PhysioNet (2023). [physionet.org/content/mimiciv](https://physionet.org/content/mimiciv/)
- **HL7 FHIR R4** — [hl7.org/fhir/R4](https://hl7.org/fhir/R4/)
- **SHAP** — Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions," NeurIPS 2017
- **BioBERT** — Lee et al., "BioBERT: a pre-trained biomedical language representation model," Bioinformatics 2020
- **Misdiagnosis statistics** — Singh H. et al., "The frequency of diagnostic errors in outpatient care," BMJ Quality & Safety 2014

---

##  Contributing

This project is in early development. Contributions welcome in:
- Clinical NLP (symptom extraction from unstructured notes)
- FHIR integration testing (against Epic/Cerner sandboxes)
- XAI evaluation methodology
- Frontend (EHR embed panel)

Open an issue to discuss before submitting a PR.

---

##  License

MIT License — see [LICENSE](LICENSE) for details.

---

> Built because Watson Health proved the AI wasn't the problem. The silence inside it was.
