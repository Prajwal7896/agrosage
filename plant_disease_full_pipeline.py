import os
import json
import shutil
import random
from collections import Counter
from multiprocessing import freeze_support

import opendatasets as od
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image

from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split

# ==========================================
# DATASET URLS
# ==========================================
GRAPE_URL = "https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset"

SUGARCANE_URL = "https://www.kaggle.com/datasets/prabhakaransoundar/sugarcane-disease-dataset"

# ==========================================
# PATHS
# ==========================================
BASE_DIR = "agri_data"

GRAPE_DIR = os.path.join(BASE_DIR, "grapes")

SUGAR_DIR = os.path.join(BASE_DIR, "sugarcane")

FINAL_DIR = "agri_dataset_final"

os.makedirs(BASE_DIR, exist_ok=True)

# ==========================================
# DOWNLOAD DATASETS
# ==========================================
def download_datasets():

    print("\n📥 DOWNLOADING DATASETS...\n")

    od.download(GRAPE_URL, data_dir=GRAPE_DIR)

    od.download(SUGARCANE_URL, data_dir=SUGAR_DIR)

# ==========================================
# RESET FINAL DATASET
# ==========================================
def prepare_dataset():

    if os.path.exists(FINAL_DIR):
        shutil.rmtree(FINAL_DIR)

    os.makedirs(FINAL_DIR)

# ==========================================
# COPY IMAGES
# ==========================================
def copy_images(src_root, crop_name):

    copied = 0

    for root, dirs, files in os.walk(src_root):

        image_files = [
            f for f in files
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(image_files) == 0:
            continue

        label = os.path.basename(root)

        if len(label) < 2:
            continue

        class_name = f"{crop_name}_{label}"

        class_dir = os.path.join(FINAL_DIR, class_name)

        os.makedirs(class_dir, exist_ok=True)

        for file in image_files:

            src_path = os.path.join(root, file)

            dst_path = os.path.join(
                class_dir,
                f"{random.randint(0,999999)}_{file}"
            )

            try:
                shutil.copy(src_path, dst_path)
                copied += 1

            except:
                pass

    print(f"✅ {crop_name}: {copied} images copied")

# ==========================================
# SEARCH & ORGANIZE DATASET
# ==========================================
def organize_dataset():

    print("\n📦 ORGANIZING DATASET...\n")

    # FIXED PATH FOR GRAPE DATASET
    grape_real_path = os.path.join(
        GRAPE_DIR,
        "new-plant-diseases-dataset",
        "New Plant Diseases Dataset(Augmented)",
        "New Plant Diseases Dataset(Augmented)",
        "train"
    )

    if os.path.exists(grape_real_path):
        copy_images(grape_real_path, "grape")
    else:
        print("❌ GRAPE TRAIN FOLDER NOT FOUND")

    sugar_real_path = os.path.join(
        SUGAR_DIR,
        "sugarcane-disease-dataset"
    )

    copy_images(sugar_real_path, "sugarcane")

    print("\n✅ DATASET READY")

    print("\n📂 FINAL CLASSES:\n")

    print(os.listdir(FINAL_DIR))

# ==========================================
# TRANSFORMS
# ==========================================
transform = transforms.Compose([

    transforms.Resize((300, 300)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(20),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ==========================================
# MODEL
# ==========================================
class AgriCNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.backbone = models.efficientnet_b3(
            weights="DEFAULT"
        )

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(

            nn.Linear(in_features, 512),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):

        x = self.backbone(x)

        x = self.classifier(x)

        return x

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":

    freeze_support()

    # DOWNLOAD
    download_datasets()

    # PREPARE
    prepare_dataset()

    # ORGANIZE
    organize_dataset()

    # DATASET
    dataset = datasets.ImageFolder(
        FINAL_DIR,
        transform=transform
    )

    print("\n🔥 TOTAL CLASSES:\n")
    print(dataset.classes)

    # SAVE CLASSES
    with open("classes.json", "w") as f:
        json.dump(dataset.classes, f)

    print("\n✅ classes.json SAVED")

    # SPLIT
    train_size = int(0.8 * len(dataset))

    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )

    # LOADERS
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    # DEVICE
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\n🚀 DEVICE: {device}")

    # MODEL
    model = AgriCNN(
        len(dataset.classes)
    ).to(device)

    # CLASS WEIGHTS
    targets = dataset.targets

    class_counts = Counter(targets)

    weights = []

    for i in range(len(dataset.classes)):
        weights.append(1.0 / class_counts[i])

    weights = torch.tensor(weights).float().to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    # OPTIMIZER
    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-5
    )

    # TRAIN
    def train():

        model.train()

        total_loss = 0

        correct = 0

        total = 0

        for images, labels in train_loader:

            images = images.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        return total_loss / len(train_loader), accuracy

    # VALIDATE
    def validate():

        model.eval()

        correct = 0

        total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)

                labels = labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

        return 100 * correct / total

    # TRAINING LOOP
    EPOCHS = 20

    best_acc = 0

    for epoch in range(EPOCHS):

        train_loss, train_acc = train()

        val_acc = validate()

        print(f"\n🔥 Epoch {epoch+1}/{EPOCHS}")

        print(f"Loss: {train_loss:.4f}")

        print(f"Train Accuracy: {train_acc:.2f}%")

        print(f"Validation Accuracy: {val_acc:.2f}%")

        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(
                model.state_dict(),
                "best_model.pth"
            )

            print("✅ BEST MODEL SAVED")

    print("\n🎉 TRAINING COMPLETE")

    print(f"\n🏆 BEST VALIDATION ACCURACY: {best_acc:.2f}%")