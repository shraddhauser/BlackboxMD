import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

DATA_ROOT = "Data"
MODEL_PATH = "models/best_densenet121.pth"
IMAGE_SIZE = 256
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_class_idx(folder_name):
    lower = folder_name.lower()
    if "adeno" in lower:
        return 0
    elif "large" in lower:
        return 1
    elif "normal" in lower:
        return 2
    elif "squamous" in lower:
        return 3
    else:
        raise ValueError(f"Unknown class directory name: {folder_name}")

class CleanImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        if not os.path.exists(root_dir):
            return
            
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if os.path.isdir(item_path):
                class_idx = get_class_idx(item)
                for fname in os.listdir(item_path):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                        self.samples.append((os.path.join(item_path, fname), class_idx))
                        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

test_dataset = CleanImageFolder(
    os.path.join(DATA_ROOT, "test"),
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint.get("class_names", [
    "adenocarcinoma", "large.cell.carcinoma", "normal", "squamous.cell.carcinoma"
])

model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(1).cpu().numpy()

        all_preds.extend(predictions)
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_preds)

print("\nTest Accuracy:", f"{accuracy * 100:.2f}%")
print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4
    )
)

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

os.makedirs("results", exist_ok=True)
np.save("results/test_labels.npy", np.array(all_labels))
np.save("results/test_predictions.npy", np.array(all_preds))

