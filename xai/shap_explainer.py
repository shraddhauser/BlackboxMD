import os
import sys
import torch
import shap
import matplotlib.pyplot as plt
import numpy as np

from xai_common import load_model, load_image

IMAGE_PATH = sys.argv[1]

model, class_names = load_model()
original_image, image_tensor = load_image(IMAGE_PATH)

with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_index = output.argmax(1).item()
    confidence = probabilities[0, predicted_index].item()

background = torch.zeros(
    (1, 3, 256, 256),
    device=image_tensor.device
)

explainer = shap.GradientExplainer(model, background)

shap_values = explainer.shap_values(
    image_tensor
)

if isinstance(shap_values, list):
    values = shap_values[predicted_index][0]
else:
    values = shap_values[0]

if isinstance(values, torch.Tensor):
    values = values.detach().cpu().numpy()

if values.ndim == 3:
    heatmap = np.abs(values).sum(axis=0)
else:
    heatmap = np.abs(values)

heatmap -= heatmap.min()

if heatmap.max() > 0:
    heatmap /= heatmap.max()

os.makedirs("results/shap", exist_ok=True)

plt.figure(figsize=(6, 6))
plt.imshow(original_image)
plt.imshow(heatmap, cmap="jet", alpha=0.45)
plt.axis("off")
plt.title(
    f"SHAP | {class_names[predicted_index]} | "
    f"{confidence * 100:.2f}%"
)
plt.tight_layout()

output_path = "results/shap/shap_result.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print("Prediction:", class_names[predicted_index])
print("Confidence:", f"{confidence * 100:.2f}%")
print("Saved:", output_path)
