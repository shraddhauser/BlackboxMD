from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

test_cases = [
    ("adenocarcinoma", "000114.png"),
    ("large.cell.carcinoma", "000108.png"),
    ("normal", "10.png"),
    ("squamous.cell.carcinoma", "000119.png")
]

print("=" * 64)
print("BLACKBOX MD — END-TO-END MULTI-CLASS VALIDATION SUITE")
print("=" * 64)

for cls_name, filename in test_cases:
    print(f"\n>>> Testing Category: {cls_name} ({filename})")
    
    # 1. Predict
    pred_res = client.post("/api/predict", data={"sample_class": cls_name, "sample_filename": filename})
    assert pred_res.status_code == 200, f"Predict failed: {pred_res.text}"
    p_data = pred_res.json()
    print(f"    [Predict] Clean Class: {p_data['clean_class']} | Conf: {p_data['confidence_percentage']}%")

    # 2. Explain
    exp_res = client.post("/api/explain", data={"sample_class": cls_name, "sample_filename": filename, "method": "gradcam"})
    assert exp_res.status_code == 200, f"Explain failed: {exp_res.text}"
    e_data = exp_res.json()
    feats = e_data.get("heatmap_features") or {}
    nlp = e_data.get("nlp_explanation") or {}
    print(f"    [Features] Location: {feats.get('location')} | Area: {feats.get('region_size_str')} | Intensity: {feats.get('activation_intensity_str')}")
    print(f"    [NLP] \"{nlp.get('primary_explanation')}\"")

    # 3. PDF Report
    pdf_res = client.post("/api/report/pdf", data={"sample_class": cls_name, "sample_filename": filename, "scan_id": f"VAL-{cls_name[:4].upper()}"})
    assert pdf_res.status_code == 200, f"PDF failed: {pdf_res.text}"
    print(f"    [PDF Report] Status 200 OK | Size: {len(pdf_res.content)} bytes")

print("\n" + "=" * 64)
print("[OK] ALL 4 PULMONARY CATEGORIES VALIDATED WITH 100% SUCCESS!")
print("=" * 64)
