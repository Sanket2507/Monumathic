"""
3. TRAIN MODEL - Monument Classification CNN Training
Trains a ResNet-based model for monument image classification.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configuration
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset_augmented"  # Use augmented dataset
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "training_logs"

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
IMG_SIZE = 128
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
EARLY_STOPPING_PATIENCE = 10

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MonumentClassifier(nn.Module):
    """Custom monument classifier based on ResNet18."""
    
    def __init__(self, num_classes, pretrained=True):
        super(MonumentClassifier, self).__init__()
        
        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=pretrained)
        
        # Modify the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


def get_data_transforms():
    """Get data transformations for training and validation."""
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Validation/Test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform


def load_dataset(dataset_path):
    """Load and split dataset."""
    
    train_transform, val_transform = get_data_transforms()
    
    # Load full dataset with train transforms
    full_dataset = datasets.ImageFolder(dataset_path, transform=train_transform)
    
    # Get class names
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    print(f"📊 Dataset Statistics:")
    print(f"   Total images: {len(full_dataset)}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Classes: {class_names[:5]}..." if len(class_names) > 5 else f"   Classes: {class_names}")
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(TRAIN_SPLIT * total_size)
    val_size = int(VAL_SPLIT * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply validation transform to val and test sets
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    print(f"   Train set: {len(train_dataset)}")
    print(f"   Validation set: {len(val_dataset)}")
    print(f"   Test set: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, class_names, num_classes


def train_epoch(model, train_loader, criterion, optimizer):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Progress
        if (batch_idx + 1) % 10 == 0:
            print(f"      Batch {batch_idx + 1}/{len(train_loader)}, "
                  f"Loss: {loss.item():.4f}, "
                  f"Acc: {100.*correct/total:.2f}%", end="\r")
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def train_model(model, train_loader, val_loader, num_epochs=EPOCHS):
    """Full training loop."""
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    print("\n🚀 Starting Training...")
    print(f"   Device: {DEVICE}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n📈 Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"   LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"   🌟 New best model! Val Acc: {best_val_acc:.2f}%")
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"\n⏹️ Early stopping triggered at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time/60:.1f} minutes")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model weights
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, history, best_val_acc


def evaluate_model(model, test_loader, class_names):
    """Evaluate model on test set with detailed metrics."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    correct = sum(p == l for p, l in zip(all_predictions, all_labels))
    total = len(all_labels)
    accuracy = 100. * correct / total
    
    print(f"\n📊 TEST RESULTS")
    print(f"   Test Accuracy: {accuracy:.2f}%")
    print(f"   Correct: {correct}/{total}")
    
    # Per-class accuracy
    class_correct = {name: 0 for name in class_names}
    class_total = {name: 0 for name in class_names}
    
    for pred, label in zip(all_predictions, all_labels):
        class_name = class_names[label]
        class_total[class_name] += 1
        if pred == label:
            class_correct[class_name] += 1
    
    print("\n   Per-class accuracy:")
    for name in class_names[:10]:  # Show first 10
        if class_total[name] > 0:
            acc = 100. * class_correct[name] / class_total[name]
            print(f"   - {name}: {acc:.1f}% ({class_correct[name]}/{class_total[name]})")
    
    if len(class_names) > 10:
        print(f"   ... and {len(class_names) - 10} more classes")
    
    return accuracy, all_predictions, all_labels


def save_model(model, class_names, history, accuracy, model_name="monument_classifier"):
    """Save trained model and metadata."""
    MODEL_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model weights
    model_path = MODEL_DIR / f"{model_name}_{timestamp}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n💾 Model saved: {model_path}")
    
    # Also save as 'best' model
    best_model_path = MODEL_DIR / "fast_monument_cnn.pth"
    torch.save(model.state_dict(), best_model_path)
    print(f"💾 Best model saved: {best_model_path}")
    
    # Save class names
    class_names_path = MODEL_DIR / "class_names.json"
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"📋 Class names saved: {class_names_path}")
    
    # Save training history
    history_path = LOG_DIR / f"training_history_{timestamp}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"📊 Training history saved: {history_path}")
    
    # Save training metadata
    metadata = {
        "model_name": model_name,
        "timestamp": timestamp,
        "num_classes": len(class_names),
        "class_names": class_names,
        "best_accuracy": accuracy,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs_trained": len(history['train_loss']),
        "device": str(DEVICE)
    }
    
    metadata_path = MODEL_DIR / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"📋 Metadata saved: {metadata_path}")
    
    return model_path


def main():
    """Main training function."""
    print("=" * 60)
    print("🏛️ MONUMENT CLASSIFIER TRAINING")
    print("=" * 60)
    
    # Check dataset
    if not DATASET_DIR.exists():
        # Try original dataset
        alt_dataset = BASE_DIR / "dataset"
        if alt_dataset.exists():
            global DATASET_DIR
            DATASET_DIR = alt_dataset
            print(f"⚠️ Using original dataset: {DATASET_DIR}")
        else:
            print(f"❌ Dataset not found: {DATASET_DIR}")
            print("   Run 1_fetch_images.py and 2_augment_data.py first.")
            return
    
    print(f"📁 Dataset: {DATASET_DIR}")
    print(f"🖥️ Device: {DEVICE}")
    
    # Load data
    train_loader, val_loader, test_loader, class_names, num_classes = load_dataset(DATASET_DIR)
    
    # Create model
    print(f"\n🔧 Creating model with {num_classes} classes...")
    model = MonumentClassifier(num_classes, pretrained=True)
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Train
    model, history, best_val_acc = train_model(model, train_loader, val_loader)
    
    # Evaluate on test set
    test_acc, _, _ = evaluate_model(model, test_loader, class_names)
    
    # Save model
    save_model(model, class_names, history, test_acc)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Monument Classifier")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--dataset", type=str, help="Path to dataset directory")
    
    args = parser.parse_args()
    
    # Update globals if provided
    if args.epochs:
        EPOCHS = args.epochs
    if args.batch_size:
        BATCH_SIZE = args.batch_size
    if args.lr:
        LEARNING_RATE = args.lr
    if args.dataset:
        DATASET_DIR = Path(args.dataset)
    
    main()
