"""
BLACKBOX MD — Heatmap Analysis & Anatomical Feature Extraction Engine
Module 5 of the BlackBoxMD-XAI Diagnostic Pipeline.

Converts raw XAI activation heatmaps (Grad-CAM, Integrated Gradients, SHAP)
into structured, quantitative clinical features that feed directly into
the Clinical NLP Explanation Module (Module 6).
"""

import numpy as np
from PIL import Image
import scipy.ndimage as ndi
from typing import Dict, Any, Tuple, Optional

IMAGE_SIZE = 256

def segment_lung_mask(image: Optional[Image.Image], target_size: Tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE)) -> Tuple[np.ndarray, int]:
    """
    Segment lung parenchyma from a chest CT scan slice.
    Uses adaptive density thresholding and thoracic geometry constraints.
    Returns a binary mask (H, W) where True indicates lung tissue, and total lung pixel count.
    """
    h, w = target_size
    if image is None:
        # Fallback to an elliptical thoracic cavity approximation
        y, x = np.ogrid[:h, :w]
        mask = (((x - w / 2) ** 2) / (0.38 * w) ** 2 + ((y - h / 2) ** 2) / (0.38 * h) ** 2) <= 1.0
        return mask, int(mask.sum())

    resized = image.resize(target_size).convert("L")
    arr = np.array(resized, dtype=np.float32)

    # 1. Torso field mask (elliptical inner rib cage boundary)
    y, x = np.ogrid[:h, :w]
    torso_mask = (((x - w / 2) ** 2) / (0.42 * w) ** 2 + ((y - h / 2) ** 2) / (0.42 * h) ** 2) <= 1.0

    # 2. Lung attenuation in CT (darker air-filled parenchyma, filtered from body tissue and scan background)
    lung_candidate = (arr < 135) & (arr > 12) & torso_mask

    # 3. Morphological cleanup using scipy.ndimage
    # Remove small noise and fill minor vessels
    struct = ndi.generate_binary_structure(2, 1)
    cleaned = ndi.binary_opening(lung_candidate, structure=struct, iterations=1)
    cleaned = ndi.binary_closing(cleaned, structure=struct, iterations=2)

    pixel_count = int(cleaned.sum())
    # Sanity guard: if segmentation is too small (<10% of image), use thoracic estimate
    if pixel_count < (h * w * 0.10):
        y, x = np.ogrid[:h, :w]
        cleaned = (((x - w / 2) ** 2) / (0.38 * w) ** 2 + ((y - h / 2) ** 2) / (0.38 * h) ** 2) <= 1.0
        pixel_count = int(cleaned.sum())

    return cleaned, pixel_count

def determine_anatomical_location(cy: float, cx: float, h: int = IMAGE_SIZE, w: int = IMAGE_SIZE) -> Dict[str, str]:
    """
    Maps image coordinates (cy, cx) to anatomical radiological pulmonary regions.
    Note standard radiological convention:
    - Left side of the image corresponds to the patient's Right Lung.
    - Right side of the image corresponds to the patient's Left Lung.
    - Upper half corresponds to the Upper / Superior lobes.
    - Lower half corresponds to the Lower / Inferior lobes.
    - Perihilar / mediastinal region is the central zone.
    """
    norm_x = cx / w  # 0.0 (image left = patient right) to 1.0 (image right = patient left)
    norm_y = cy / h  # 0.0 (anterior / upper) to 1.0 (posterior / lower)

    # Check for central hilar / mediastinal involvement
    is_central_x = 0.40 <= norm_x <= 0.60
    is_central_y = 0.35 <= norm_y <= 0.65

    if is_central_x and is_central_y:
        if norm_x < 0.48:
            location = "Right Hilar Region"
            quadrant = "Right Central"
        elif norm_x > 0.52:
            location = "Left Hilar Region"
            quadrant = "Left Central"
        else:
            location = "Central Mediastinum / Hilum"
            quadrant = "Central"
    else:
        # Lateral determination (Radiological: Left is Patient Right, Right is Patient Left)
        side = "Right" if norm_x < 0.50 else "Left"
        vertical = "Upper" if norm_y < 0.50 else "Lower"
        
        location = f"{vertical} {side} Lung"
        quadrant = f"{side} {vertical}"

    return {
        "location": location,
        "quadrant": quadrant,
        "normalized_coords": f"x={norm_x:.2f}, y={norm_y:.2f}"
    }

def calculate_activation_concentration(mask: np.ndarray, cy: float, cx: float) -> Tuple[str, float]:
    """
    Measures the spatial dispersion of activations around their centroid.
    A compact cluster indicates a focal lesion (high concentration);
    a widely dispersed pattern indicates diffuse ground-glass or multi-focal activation.
    """
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0:
        return "Low (Diffuse)", 0.0

    std_y = np.std(y_indices)
    std_x = np.std(x_indices)
    radius = float(np.sqrt(std_x**2 + std_y**2))

    # Normalize radius by image dimension
    dispersion = radius / float(mask.shape[0])

    if dispersion < 0.12:
        return "High (Focal)", round(dispersion, 3)
    elif dispersion < 0.22:
        return "Moderate (Localized)", round(dispersion, 3)
    else:
        return "Low (Diffuse)", round(dispersion, 3)

def extract_heatmap_features(
    heatmap: np.ndarray,
    original_image: Optional[Image.Image] = None,
    threshold_ratio: float = 0.40
) -> Dict[str, Any]:
    """
    Main extraction pipeline:
    Analyzes a 2D normalized activation heatmap (H, W) and extracts:
    - Dominant anatomical location (e.g. 'Upper Right Lung')
    - Activation Level ('High', 'Moderate', 'Mild')
    - Region Size (% of lung area)
    - Activation Intensity (Peak and mean activation)
    - Activation Concentration ('High (Focal)', 'Moderate', 'Diffuse')
    - Bounding Box & Centroid coordinates
    """
    if heatmap.ndim != 2:
        heatmap = np.squeeze(heatmap)
    if heatmap.ndim != 2:
        raise ValueError(f"Heatmap must be 2D array, got shape: {heatmap.shape}")

    h, w = heatmap.shape

    # 1. Normalize heatmap to [0, 1] if not already
    h_min, h_max = float(heatmap.min()), float(heatmap.max())
    if h_max > h_min:
        norm_map = (heatmap - h_min) / (h_max - h_min)
    else:
        norm_map = np.zeros_like(heatmap)

    peak_intensity = float(h_max if h_max <= 1.0 else 1.0)
    
    # 2. Segment lung parenchyma to establish relative area baseline
    lung_mask, lung_pixels = segment_lung_mask(original_image, target_size=(h, w))

    # 3. Identify activated lesion region
    # A pixel is considered significantly activated if it exceeds threshold_ratio of peak
    effective_thresh = max(0.35, threshold_ratio)
    activated_mask = norm_map >= effective_thresh
    
    # Restrict activation to thoracic / lung boundaries for clinical realism
    clinically_activated = activated_mask & lung_mask
    if clinically_activated.sum() == 0:
        # Fallback if activations are outside segmented boundary
        clinically_activated = activated_mask

    activated_pixels = int(clinically_activated.sum())

    # 4. Region Size (% of lung area)
    if lung_pixels > 0:
        region_size_pct = round((activated_pixels / lung_pixels) * 100.0, 1)
        # Cap region size realistically for solitary nodules/masses (1% to 65%)
        region_size_pct = max(1.0, min(region_size_pct, 65.0))
    else:
        region_size_pct = round((activated_pixels / (h * w)) * 100.0, 1)

    # 5. Centroid & Anatomical Location
    # Weight coordinates by squared activation to focus on the epicenter
    weights = (norm_map ** 2) * clinically_activated.astype(np.float32)
    if weights.sum() > 0:
        cy, cx = ndi.center_of_mass(weights)
        cy, cx = float(cy), float(cx)
    else:
        cy, cx = float(h / 2), float(w / 2)

    anatomical_info = determine_anatomical_location(cy, cx, h=h, w=w)

    # 6. Activation Intensity & Level
    if activated_pixels > 0:
        mean_intensity = float(np.mean(norm_map[clinically_activated]))
    else:
        mean_intensity = float(np.mean(norm_map))

    if peak_intensity >= 0.75:
        activation_level = "High"
    elif peak_intensity >= 0.50:
        activation_level = "Moderate"
    else:
        activation_level = "Low"

    # 7. Activation Concentration
    concentration, dispersion_score = calculate_activation_concentration(clinically_activated, cy, cx)

    # 8. Bounding Box around the primary focus
    y_indices, x_indices = np.where(clinically_activated)
    if len(y_indices) > 0:
        ymin, ymax = int(y_indices.min()), int(y_indices.max())
        xmin, xmax = int(x_indices.min()), int(x_indices.max())
    else:
        ymin, xmin, ymax, xmax = int(cy - 20), int(cx - 20), int(cy + 20), int(cx + 20)

    bbox = {
        "ymin": max(0, ymin),
        "xmin": max(0, xmin),
        "ymax": min(h, ymax),
        "xmax": min(w, xmax),
        "width": max(1, xmax - xmin),
        "height": max(1, ymax - ymin)
    }

    return {
        "location": anatomical_info["location"],
        "activation_level": activation_level,
        "region_size_pct": region_size_pct,
        "region_size_str": f"{region_size_pct:.1f}% of lung area",
        "activation_intensity": round(peak_intensity, 2),
        "activation_intensity_str": f"{peak_intensity:.2f} ({activation_level})",
        "mean_intensity": round(mean_intensity, 2),
        "concentration": concentration,
        "dispersion_score": dispersion_score,
        "center_of_mass": {"y": round(cy, 1), "x": round(cx, 1)},
        "bounding_box": bbox,
        "anatomical_quadrant": anatomical_info["quadrant"],
        "lung_pixels": lung_pixels,
        "activated_pixels": activated_pixels
    }


if __name__ == "__main__":
    print("BLACKBOX MD — Feature Extractor Self-Test")
    # Generate synthetic 256x256 test heatmap with focus in Upper Right (Image Left, Top)
    synthetic_heatmap = np.zeros((256, 256), dtype=np.float32)
    y, x = np.ogrid[:256, :256]
    # Center at (y=75, x=75) -> Upper Right Lung
    dist = np.sqrt((x - 75)**2 + (y - 75)**2)
    synthetic_heatmap = np.exp(-(dist**2) / (2 * (25**2)))

    features = extract_heatmap_features(synthetic_heatmap)
    print("\nExtracted Synthetic Heatmap Features:")
    for k, v in features.items():
        print(f"  {k}: {v}")
