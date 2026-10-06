import os
import io
import json
import base64
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from xai.xai_engine import (
    load_model,
    preprocess_image,
    predict,
    run_gradcam,
    run_integrated_gradients,
    run_shap,
    device,
    MODEL_PATH,
    DEFAULT_CLASSES
)
from xai.feature_extractor import extract_heatmap_features
from xai.nlp_explainer import generate_clinical_explanation, clean_class_name
from xai.report_generator import generate_pdf_report

app = FastAPI(
    title="BLACKBOX MD — XAI Backend API",
    description="Comparative Explainable AI API for Pulmonary Oncology Diagnostics",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
STATIC_DIR = BASE_DIR / "static"
RESULTS_DIR = BASE_DIR / "results"

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def find_sample_image_path(sample_class: str, sample_filename: str) -> Optional[Path]:
    test_dir = DATA_DIR / "test"
    if not test_dir.exists():
        return None
    direct = test_dir / sample_class / sample_filename
    if direct.exists():
        return direct
    
    for folder in test_dir.iterdir():
        if folder.is_dir():
            if folder.name == sample_class or folder.name.startswith(sample_class) or sample_class.startswith(folder.name):
                candidate = folder / sample_filename
                if candidate.exists():
                    return candidate
    return None

FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    if (FRONTEND_DIST / "index.html").exists():
        return FileResponse(FRONTEND_DIST / "index.html")
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h2>BLACKBOX MD API Server running.</h2>")

@app.get("/api/status")
async def get_status():
    model_exists = os.path.exists(MODEL_PATH)
    try:
        model, classes = load_model()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        classes = DEFAULT_CLASSES

    clean_classes = [clean_class_name(c) for c in classes]

    return {
        "status": "online",
        "device": str(device),
        "model": "DenseNet-121",
        "model_checkpoint_exists": model_exists,
        "model_loaded": model_loaded,
        "num_classes": len(classes),
        "classes": classes,
        "clean_classes": clean_classes,
        "capabilities": {
            "xai_methods": ["Grad-CAM", "Integrated Gradients", "SHAP"],
            "feature_extraction": True,
            "nlp_explanation": True,
            "pdf_export": True
        }
    }

@app.get("/api/samples")
async def get_samples():
    test_dir = DATA_DIR / "test"
    if not test_dir.exists():
        return {"samples": {}}

    samples = {}
    for class_folder in test_dir.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            files = [
                f.name for f in class_folder.glob("*.png")
            ] + [
                f.name for f in class_folder.glob("*.jpg")
            ] + [
                f.name for f in class_folder.glob("*.jpeg")
            ]
            samples[class_name] = files[:12]  # Return up to 12 sample files per class

    return {"samples": samples}

@app.get("/api/sample-image/{class_name}/{filename}")
async def get_sample_image(class_name: str, filename: str):
    image_path = find_sample_image_path(class_name, filename)
    if not image_path:
        raise HTTPException(status_code=404, detail="Sample image not found")
    return FileResponse(image_path)

@app.post("/api/predict")
async def handle_predict(
    file: Optional[UploadFile] = File(None),
    sample_class: Optional[str] = Form(None),
    sample_filename: Optional[str] = Form(None)
):
    try:
        if file is not None:
            contents = await file.read()
            original_image, image_tensor = preprocess_image(contents)
            source = f"Uploaded file: {file.filename}"
        elif sample_class and sample_filename:
            image_path = find_sample_image_path(sample_class, sample_filename)
            if not image_path:
                raise HTTPException(status_code=404, detail="Sample image file not found")
            original_image, image_tensor = preprocess_image(image_path)
            source = f"Sample: {sample_class}/{sample_filename}"
        else:
            raise HTTPException(status_code=400, detail="Must provide an uploaded image file or sample_class and sample_filename")

        model, class_names = load_model()
        predicted_index, confidence, probabilities = predict(model, image_tensor, class_names)

        # Convert original image to base64 for preview
        buf = io.BytesIO()
        original_image.resize((256, 256)).save(buf, format="PNG")
        orig_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        raw_class = class_names[predicted_index]
        clean_class = clean_class_name(raw_class)

        # Formatted probability dictionary
        clean_probabilities = {
            clean_class_name(cls): prob for cls, prob in probabilities.items()
        }

        return {
            "source": source,
            "predicted_class": raw_class,
            "clean_class": clean_class,
            "predicted_index": predicted_index,
            "confidence": confidence,
            "confidence_percentage": round(confidence * 100, 1),
            "probabilities": probabilities,
            "clean_probabilities": clean_probabilities,
            "original_image_base64": f"data:image/png;base64,{orig_b64}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/explain")
async def handle_explain(
    file: Optional[UploadFile] = File(None),
    sample_class: Optional[str] = Form(None),
    sample_filename: Optional[str] = Form(None),
    method: str = Form("all") # "gradcam", "integrated_gradients", "shap", or "all"
):
    """
    Unified Explainability Pipeline:
    Runs visual XAI (Grad-CAM, IG, SHAP), extracts quantitative spatial features
    (Step 5), and generates clinical NLP narrative (Step 6).
    """
    try:
        if file is not None:
            contents = await file.read()
            original_image, image_tensor = preprocess_image(contents)
            source = f"Uploaded file: {file.filename}"
        elif sample_class and sample_filename:
            image_path = find_sample_image_path(sample_class, sample_filename)
            if not image_path:
                raise HTTPException(status_code=404, detail="Sample image file not found")
            original_image, image_tensor = preprocess_image(image_path)
            source = f"Sample: {sample_class}/{sample_filename}"
        else:
            raise HTTPException(status_code=400, detail="Must provide an image file or sample identifiers")

        model, class_names = load_model()
        predicted_index, confidence, probabilities = predict(model, image_tensor, class_names)
        raw_class = class_names[predicted_index]
        clean_class = clean_class_name(raw_class)

        explanations = {}
        primary_features = None

        method_list = [m.strip().lower() for m in method.split(",")]

        if "gradcam" in method_list or "all" in method_list:
            gcam_res = run_gradcam(model, image_tensor, predicted_index, original_image)
            explanations["gradcam"] = gcam_res
            if gcam_res and gcam_res.get("features"):
                primary_features = gcam_res["features"]

        if "integrated_gradients" in method_list or "ig" in method_list or "all" in method_list:
            ig_res = run_integrated_gradients(model, image_tensor, predicted_index, original_image)
            explanations["integrated_gradients"] = ig_res
            if not primary_features and ig_res and ig_res.get("features"):
                primary_features = ig_res["features"]

        if "shap" in method_list or "all" in method_list:
            shap_res = run_shap(model, image_tensor, predicted_index, original_image)
            explanations["shap"] = shap_res
            if not primary_features and shap_res and shap_res.get("features"):
                primary_features = shap_res["features"]

        # Step 6: Generate Clinical NLP Explanation
        nlp_explanation = generate_clinical_explanation(
            predicted_class=raw_class,
            confidence=confidence,
            features=primary_features,
            probabilities=probabilities
        )

        # Original image base64
        buf = io.BytesIO()
        original_image.resize((256, 256)).save(buf, format="PNG")
        orig_b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        return {
            "source": source,
            "predicted_class": raw_class,
            "clean_class": clean_class,
            "confidence": confidence,
            "confidence_percentage": round(confidence * 100, 1),
            "probabilities": probabilities,
            "original_image_base64": orig_b64,
            "explanations": explanations,
            "heatmap_features": primary_features,
            "nlp_explanation": nlp_explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/report/pdf")
async def handle_generate_pdf_report(
    file: Optional[UploadFile] = File(None),
    sample_class: Optional[str] = Form(None),
    sample_filename: Optional[str] = Form(None),
    scan_id: Optional[str] = Form(None),
    patient_id: Optional[str] = Form(None)
):
    """
    Step 7 Report Generator:
    Generates and downloads a complete Medical Diagnostic PDF report.
    """
    try:
        if file is not None:
            contents = await file.read()
            original_image, image_tensor = preprocess_image(contents)
            scan_ref = scan_id or f"CT-UPLOAD-{file.filename[:8]}"
        elif sample_class and sample_filename:
            image_path = find_sample_image_path(sample_class, sample_filename)
            if not image_path:
                raise HTTPException(status_code=404, detail="Sample image file not found")
            original_image, image_tensor = preprocess_image(image_path)
            scan_ref = scan_id or f"CT-{sample_class[:6]}-{sample_filename.split('.')[0]}"
        else:
            raise HTTPException(status_code=400, detail="Must provide an image file or sample identifiers")

        model, class_names = load_model()
        predicted_index, confidence, probabilities = predict(model, image_tensor, class_names)
        raw_class = class_names[predicted_index]

        # Generate Grad-CAM for the report visual
        gcam = run_gradcam(model, image_tensor, predicted_index, original_image)
        gradcam_overlay = gcam["overlay_base64"] if gcam else original_image
        features = gcam.get("features", {}) if gcam else {}

        # Generate NLP Narrative
        nlp_data = generate_clinical_explanation(
            predicted_class=raw_class,
            confidence=confidence,
            features=features,
            probabilities=probabilities
        )

        pred_data = {
            "predicted_class": raw_class,
            "confidence": confidence,
            "probabilities": probabilities
        }

        # Build PDF
        pdf_bytes = generate_pdf_report(
            original_image=original_image,
            gradcam_image=gradcam_overlay,
            prediction_data=pred_data,
            features=features,
            nlp_data=nlp_data,
            scan_id=scan_ref,
            patient_id=patient_id or "ANONYMIZED-STUDY"
        )

        filename = f"BlackBoxMD_Report_{scan_ref}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation error: {str(e)}")

@app.post("/api/report/export-custom-pdf")
async def handle_export_custom_pdf(payload: Dict[str, Any] = Body(...)):
    """
    Direct PDF exporter:
    Accepts pre-computed JSON state (images, prediction, features, nlp)
    and instantly renders a downloadable PDF without recomputing.
    """
    try:
        original_image = payload.get("original_image_base64", "")
        gradcam_image = payload.get("gradcam_image_base64", original_image)
        prediction_data = payload.get("prediction_data", {})
        features = payload.get("features", {})
        nlp_data = payload.get("nlp_data", {})
        scan_id = payload.get("scan_id", "CT-STUDY-CUSTOM")
        patient_id = payload.get("patient_id", "ANONYMIZED-STUDY")

        pdf_bytes = generate_pdf_report(
            original_image=original_image,
            gradcam_image=gradcam_image,
            prediction_data=prediction_data,
            features=features,
            nlp_data=nlp_data,
            scan_id=scan_id,
            patient_id=patient_id
        )

        filename = f"BlackBoxMD_Report_{scan_id}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Custom PDF export error: {str(e)}")

@app.get("/api/metrics")
async def get_metrics():
    history_path = BASE_DIR / "models" / "training_history.npy"
    labels_path = RESULTS_DIR / "test_labels.npy"
    preds_path = RESULTS_DIR / "test_predictions.npy"

    history = None
    if history_path.exists():
        try:
            history = np.load(history_path, allow_pickle=True).item()
        except Exception as e:
            print("Error loading training history:", e)

    test_results = None
    if labels_path.exists() and preds_path.exists():
        try:
            labels = np.load(labels_path)
            preds = np.load(preds_path)
            from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
            acc = float(accuracy_score(labels, preds))
            cm = confusion_matrix(labels, preds).tolist()
            _, class_names = load_model()
            report = classification_report(labels, preds, target_names=class_names, output_dict=True)
            test_results = {
                "accuracy": acc,
                "confusion_matrix": cm,
                "classification_report": report
            }
        except Exception as e:
            print("Error loading test results:", e)

    return {
        "history": history,
        "test_results": test_results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


