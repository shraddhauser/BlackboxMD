# 🩺 BLACKBOX MD
### Explainable AI Second Opinion System for Misdiagnosis Prevention

> *"The doctor isn't wrong. The AI just wants to show its work."*

## Problem

Millions of diagnostic errors happen each year because clinical AI is often a black box. Doctors need transparent reasoning, not silent predictions.

## Solution

BLACKBOX MD is an advisory layer that listens to EHR diagnosis events, compares them against patient data and historical cases, and raises a quiet flag with a plain-language explanation when something looks inconsistent.

- Keeps the doctor in control
- Never overrides clinical decisions
- Explains why the flag was raised
- Uses FHIR R4 for EHR interoperability

## Architecture

- `ingestion/` — FHIR data fetch and feature assembly
- `model/` — XGBoost + clinical NLP diagnosis classifier
- `explainer/` — SHAP/LIME explainability and sentence generation
- `alerts/` — severity scoring, EHR notification, audit logging
- `api/` — FastAPI analysis endpoints
- `dashboard/` — embedded UI panel for flags and explanations

## Getting Started

```bash
git clone https://github.com/yourusername/blackbox-md.git
cd blackbox-md
pip install -r requirements.txt
docker-compose up --build
```

### Example

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"demo-001","fhir_server":"https://hapi.fhir.org/baseR4"}'
```

## Status

- FHIR R4 ingestion: ✅
- XGBoost classifier: ✅
- SHAP explainability: ✅
- Plain-language generator: in progress
- EHR alert push: in progress
- Dashboard embed: planned

## Contributing

Contributions welcome for clinical NLP, FHIR integration, XAI, and frontend work. Open an issue first.


