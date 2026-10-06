import os
import sys
import torch
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

from xai_common import load_model, load_image

IMAGE_PATH = sys.argv[1]

model, class_names = load_model()
original_image, image_tensor = load_image(IMAGE_PATH)

model.zero_grad()

with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_index = output.argmax(1).item()
    confidence = probabilities[0, predicted_index].item()

baseline = torch.zeros_like(image_tensor)

ig = IntegratedGradients(model)

attributions = ig.attribute(
    image_tensor,
    baselines=baseline,
    target=predicted_index,
    n_steps=50
)

attributions = attributions[0].detach().cpu()

heatmap = attributions.abs().sum(dim=0).numpy()

heatmap -= heatmap.min()

if heatmap.max() > 0:
    heatmap /= heatmap.max()

os.makedirs("results/integrated_gradients", exist_ok=True)

plt.figure(figsize=(6, 6))
plt.imshow(original_image)
plt.imshow(heatmap, cmap="jet", alpha=0.45)
plt.axis("off")
plt.title(
    f"Integrated Gradients | {class_names[predicted_index]} | "
    f"{confidence * 100:.2f}%"
)
plt.tight_layout()

output_path = "results/integrated_gradients/integrated_gradients_result.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print("Prediction:", class_names[predicted_index])
print("Confidence:", f"{confidence * 100:.2f}%")
print("Saved:", output_path)
