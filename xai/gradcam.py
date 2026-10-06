import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

from xai_common import load_model, load_image, predict, device

IMAGE_PATH = sys.argv[1]

model, class_names = load_model()
original_image, image_tensor = load_image(IMAGE_PATH)

predicted_index, confidence, probabilities = predict(
    model, image_tensor, class_names
)

features = None
gradients = None

def forward_hook(module, input, output):
    global features
    features = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

target_layer = model.features.denseblock4

forward_handle = target_layer.register_forward_hook(forward_hook)
backward_handle = target_layer.register_full_backward_hook(backward_hook)

model.zero_grad()

output = model(image_tensor)
score = output[0, predicted_index]
score.backward()

forward_handle.remove()
backward_handle.remove()

weights = gradients.mean(dim=(2, 3), keepdim=True)
cam = (weights * features).sum(dim=1, keepdim=True)
cam = torch.relu(cam)

cam = torch.nn.functional.interpolate(
    cam,
    size=(IMAGE_SIZE := 256, 256),
    mode="bilinear",
    align_corners=False
)

cam = cam[0, 0].detach().cpu().numpy()
cam -= cam.min()

if cam.max() > 0:
    cam /= cam.max()

os.makedirs("results/gradcam", exist_ok=True)

plt.figure(figsize=(6, 6))
plt.imshow(original_image)
plt.imshow(cam, cmap="jet", alpha=0.45)
plt.axis("off")
plt.title(
    f"Grad-CAM | {class_names[predicted_index]} | "
    f"{confidence * 100:.2f}%"
)
plt.tight_layout()

output_path = "results/gradcam/gradcam_result.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print("Prediction:", class_names[predicted_index])
print("Confidence:", f"{confidence * 100:.2f}%")
print("Saved:", output_path)
