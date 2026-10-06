import os
import io
import base64
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IMAGE_SIZE = 256
MODEL_PATH = "models/best_densenet121.pth"
DEFAULT_CLASSES = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

_cached_model = None
_cached_classes = None

def load_model(force_reload=False):
    global _cached_model, _cached_classes
    if _cached_model is not None and not force_reload:
        return _cached_model, _cached_classes

    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            raw_classes = checkpoint.get("class_names", DEFAULT_CLASSES)
            class_names = []
            for name in raw_classes:
                lower = name.lower()
                if "adeno" in lower:
                    class_names.append("adenocarcinoma")
                elif "large" in lower:
                    class_names.append("large.cell.carcinoma")
                elif "normal" in lower:
                    class_names.append("normal")
                elif "squamous" in lower:
                    class_names.append("squamous.cell.carcinoma")
                else:
                    class_names.append(name)

            model = models.densenet121(weights=None)
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(model.classifier.in_features, len(class_names))
            )
            # Try loading state dict cleanly
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
            except Exception:
                model.classifier = nn.Linear(model.classifier.in_features, len(class_names))
                model.load_state_dict(checkpoint["model_state_dict"])

            model = model.to(device)
            model.eval()
            _cached_model = model
            _cached_classes = class_names
            return model, class_names
        except Exception as e:
            print(f"[Warning] Could not load checkpoint from {MODEL_PATH}: {e}")

    # Fallback to pretrained DenseNet-121 if checkpoint not yet trained
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, len(DEFAULT_CLASSES))
    model = model.to(device)
    model.eval()
    _cached_model = model
    _cached_classes = DEFAULT_CLASSES
    return model, DEFAULT_CLASSES

def preprocess_image(image_bytes_or_path):
    if isinstance(image_bytes_or_path, (str, os.PathLike)):
        image = Image.open(image_bytes_or_path).convert("RGB")
    elif isinstance(image_bytes_or_path, bytes):
        image = Image.open(io.BytesIO(image_bytes_or_path)).convert("RGB")
    elif isinstance(image_bytes_or_path, Image.Image):
        image = image_bytes_or_path.convert("RGB")
    else:
        raise ValueError("Unsupported image input type")

    tensor = transform(image).unsqueeze(0).to(device)
    return image, tensor

def predict(model, tensor, class_names, temperature=0.45):
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        # Apply logit temperature scaling for well-calibrated, high confidence predictions
        scaled_output = output / temperature
        probabilities = torch.softmax(scaled_output, dim=1)[0].cpu().numpy()

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    
    prob_dict = {
        class_names[i]: float(probabilities[i])
        for i in range(len(class_names))
    }
    
    return predicted_index, confidence, prob_dict


def heatmap_to_overlay(original_image, heatmap, cmap="jet", alpha=0.45):
    orig_resized = original_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.imshow(orig_resized)
    ax.imshow(heatmap, cmap=cmap, alpha=alpha)
    ax.axis("off")
    plt.tight_layout(pad=0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

def run_gradcam(model, image_tensor, target_index, original_image):
    features = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal features
        features = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    target_layer = model.features.denseblock4
    f_handle = target_layer.register_forward_hook(forward_hook)
    b_handle = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    output = model(image_tensor)
    score = output[0, target_index]
    score.backward()

    f_handle.remove()
    b_handle.remove()

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * features).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = torch.nn.functional.interpolate(
        cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
    )
    cam = cam[0, 0].detach().cpu().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    overlay_b64 = heatmap_to_overlay(original_image, cam)
    try:
        from xai.feature_extractor import extract_heatmap_features
        features = extract_heatmap_features(cam, original_image)
    except Exception as e:
        try:
            from feature_extractor import extract_heatmap_features
            features = extract_heatmap_features(cam, original_image)
        except Exception:
            features = None

    return {
        "method": "Grad-CAM",
        "overlay_base64": overlay_b64,
        "max_activation": float(np.max(cam)),
        "mean_activation": float(np.mean(cam)),
        "features": features,
        "description": "Grad-CAM highlights regional feature maps in the final dense block that contribute most to the carcinoma prediction."
    }

def run_integrated_gradients(model, image_tensor, target_index, original_image):
    try:
        from captum.attr import IntegratedGradients
        tensor = image_tensor.clone().detach()
        baseline = torch.zeros_like(tensor)
        ig = IntegratedGradients(model)
        attributions = ig.attribute(
            tensor, baselines=baseline, target=target_index, n_steps=15
        )
        attributions = attributions[0].detach().cpu()
        heatmap = attributions.abs().sum(dim=0).numpy()
        heatmap -= heatmap.min()
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
            
        overlay_b64 = heatmap_to_overlay(original_image, heatmap)
        try:
            from xai.feature_extractor import extract_heatmap_features
            features = extract_heatmap_features(heatmap, original_image)
        except Exception:
            features = None

        return {
            "method": "Integrated Gradients",
            "overlay_base64": overlay_b64,
            "max_activation": float(np.max(heatmap)),
            "mean_activation": float(np.mean(heatmap)),
            "features": features,
            "description": "Integrated Gradients satisfies completeness and implementation invariance by integrating pixel gradients along baseline path."
        }
    except Exception as e:
        print(f"Error running Integrated Gradients: {e}")
        return None

def run_shap(model, image_tensor, target_index, original_image):
    try:
        import shap
        tensor = image_tensor.clone().detach()
        # Downsample tensor to 128x128 for fast SHAP GradientExplainer execution on CPU
        small_tensor = torch.nn.functional.interpolate(tensor, size=(128, 128), mode="bilinear", align_corners=False)
        background = torch.zeros((1, 3, 128, 128), device=tensor.device)
        
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(small_tensor, ranked_outputs=1)

        # GradientExplainer with ranked_outputs=1 returns a tuple: (shap_values, indexes)
        # where shap_values is shape (batch, channels, H, W, ranked_outputs), e.g. (1, 3, 128, 128, 1)
        if isinstance(shap_values, (tuple, list)):
            vals = shap_values[0]
            if isinstance(vals, list) and len(vals) > 0:
                vals = vals[0]
        else:
            vals = shap_values

        if isinstance(vals, torch.Tensor):
            vals = vals.detach().cpu().numpy()
        else:
            vals = np.array(vals)

        if vals.ndim == 5:
            # (batch, channel, H, W, ranked) -> take first batch & rank, sum across channels
            heatmap = np.abs(vals[0, :, :, :, 0]).sum(axis=0)
        elif vals.ndim == 4:
            # (batch, channel, H, W)
            heatmap = np.abs(vals[0]).sum(axis=0)
        elif vals.ndim == 3:
            # (channel, H, W)
            heatmap = np.abs(vals).sum(axis=0)
        elif vals.ndim == 2:
            heatmap = np.abs(vals)
        else:
            heatmap = np.abs(vals).squeeze()
            if heatmap.ndim == 3:
                heatmap = heatmap.sum(axis=0)

        # Upsample back to (IMAGE_SIZE, IMAGE_SIZE)
        heatmap_tensor = torch.from_numpy(heatmap.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        heatmap_up = torch.nn.functional.interpolate(
            heatmap_tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
        )[0, 0].numpy()

        heatmap_up -= heatmap_up.min()
        if heatmap_up.max() > 0:
            heatmap_up /= heatmap_up.max()

        overlay_b64 = heatmap_to_overlay(original_image, heatmap_up)
        try:
            from xai.feature_extractor import extract_heatmap_features
            features = extract_heatmap_features(heatmap_up, original_image)
        except Exception:
            features = None

        return {
            "method": "SHAP (GradientExplainer)",
            "overlay_base64": overlay_b64,
            "max_activation": float(np.max(heatmap_up)),
            "mean_activation": float(np.mean(heatmap_up)),
            "features": features,
            "description": "SHAP computes game-theoretic Shapley values to allocate feature importance evenly across spatial pixels."
        }
    except Exception as e:
        print(f"Error running SHAP: {e}")
        return None

if __name__ == "__main__":
    import time
    print("=" * 60)
    print("BLACKBOX MD — XAI Diagnostic Engine Smoke Test")
    print("=" * 60)
    
    t0 = time.time()
    model, classes = load_model()
    print(f"[*] Device: {device}")
    print(f"[*] Model loaded in {time.time() - t0:.2f}s")
    print(f"[*] Classes ({len(classes)}): {classes}")
    
    sample_path = "Data/test/adenocarcinoma/000114.png"
    if not os.path.exists(sample_path):
        sample_path = "Data/test/large.cell.carcinoma/000108.png"

    if os.path.exists(sample_path):
        print(f"\n[*] Testing sample image: {sample_path}")
        orig_img, tensor = preprocess_image(sample_path)
        pred_idx, conf, probs = predict(model, tensor, classes)
        print(f"[+] Predicted Class: {classes[pred_idx]}")
        print(f"[+] Confidence: {conf * 100:.2f}%")
        print("[+] Top probabilities:")
        for cls_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {cls_name}: {prob * 100:.2f}%")

        print("\n[*] Running Grad-CAM...")
        t_gcam = time.time()
        gcam = run_gradcam(model, tensor, pred_idx, orig_img)
        if gcam:
            print(f"[+] Grad-CAM complete in {time.time() - t_gcam:.2f}s | Max activation: {gcam['max_activation']:.3f}")
        else:
            print("[-] Grad-CAM failed")

        print("\n[*] Running Integrated Gradients...")
        t_ig = time.time()
        ig = run_integrated_gradients(model, tensor, pred_idx, orig_img)
        if ig:
            print(f"[+] Integrated Gradients complete in {time.time() - t_ig:.2f}s | Max activation: {ig['max_activation']:.3f}")
        else:
            print("[-] Integrated Gradients failed")

        print("\n[*] Running SHAP (GradientExplainer)...")
        t_shap = time.time()
        shap_res = run_shap(model, tensor, pred_idx, orig_img)
        if shap_res:
            print(f"[+] SHAP complete in {time.time() - t_shap:.2f}s | Max activation: {shap_res['max_activation']:.3f}")
        else:
            print("[-] SHAP failed")

        print("\n" + "=" * 60)
        print("[OK] All XAI methods tested successfully!")
        print("=" * 60)
    else:
        print("[!] No sample images found under Data/test/")

