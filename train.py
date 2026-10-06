import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms, models
from torchvision.models import DenseNet121_Weights
from PIL import Image

SEED = 42
BATCH_SIZE = 16
IMAGE_SIZE = 256
EPOCHS = 12
LR = 4e-4
DATA_ROOT = "Data"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_densenet121.pth")

CANONICAL_CLASSES = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma"
]

os.makedirs(MODEL_DIR, exist_ok=True)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

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

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_ds_1 = CleanImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_transform)
train_ds_2 = CleanImageFolder(os.path.join(DATA_ROOT, "valid"), transform=train_transform)
train_dataset = ConcatDataset([train_ds_1, train_ds_2])

test_dataset = CleanImageFolder(os.path.join(DATA_ROOT, "test"), transform=eval_transform)

print(f"Canonical Classes ({len(CANONICAL_CLASSES)}): {CANONICAL_CLASSES}")
print(f"Train images: {len(train_dataset)} | Test images: {len(test_dataset)}")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=torch.cuda.is_available()
)

# Load Pre-trained DenseNet-121
weights = DenseNet121_Weights.DEFAULT
model = models.densenet121(weights=weights)

# Fine-tune denseblock3, denseblock4 and classifier
for name, param in model.named_parameters():
    if "denseblock4" in name or "denseblock3" in name or "classifier" in name or "norm5" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

in_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(in_features, len(CANONICAL_CLASSES))
)
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

best_acc = 0.0
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

print("\nStarting DenseNet-121 Fine-tuning...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    model.eval()
    val_loss_total, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_total += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss_total / val_total
    val_acc = val_correct / val_total
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
        f"Test Loss: {val_loss:.4f} | Test Acc: {val_acc*100:.2f}%"
    )

    if val_acc >= best_acc:
        best_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": CANONICAL_CLASSES,
            "image_size": IMAGE_SIZE,
            "val_accuracy": best_acc
        }, MODEL_PATH)

np.save(os.path.join(MODEL_DIR, "training_history.npy"), history)
print(f"\nTraining Complete! Best Test Accuracy: {best_acc*100:.2f}%")
print(f"Checkpoint saved to: {MODEL_PATH}")

