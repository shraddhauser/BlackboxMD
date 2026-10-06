import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

IMAGE_SIZE = 256
MODEL_PATH = "models/best_densenet121.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    class_names = checkpoint["class_names"]

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(
        model.classifier.in_features,
        len(class_names)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, class_names

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    return image, tensor

def predict(model, tensor, class_names):
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)

    predicted_index = probabilities.argmax(1).item()
    confidence = probabilities[0, predicted_index].item()

    return predicted_index, confidence, probabilities
