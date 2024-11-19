import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from torch.multiprocessing import freeze_support
from datetime import datetime

def check_dataset_structure(folder_path):
    print(f"Checking dataset structure in {folder_path}")
    if not os.path.exists(folder_path):
        print(f"Error: {folder_path} does not exist!")
        return False
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    pokemon_classes = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    
    if not pokemon_classes:
        print(f"Error: No class folders found in {folder_path}")
        return False
    
    total_images = 0
    print("\nFound classes:")
    for pokemon in pokemon_classes:
        class_path = os.path.join(folder_path, pokemon)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(valid_extensions)]
        total_images += len(images)
        print(f"{pokemon}: {len(images)} images")
    
    print(f"\nTotal classes: {len(pokemon_classes)}")
    print(f"Total images: {total_images}")
    
    return total_images > 0

def train_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define paths
    train_path = 'C:/Users/Joab/Documents/PokeGuess/train'
    val_path = 'C:/Users/Joab/Documents/PokeGuess/val'

    # Check dataset structure
    print("\nChecking training dataset:")
    train_valid = check_dataset_structure(train_path)
    print("\nChecking validation dataset:")
    val_valid = check_dataset_structure(val_path)

    if not (train_valid and val_valid):
        print("\nDataset structure check failed. Please fix the issues before training.")
        return

    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    print("\nLoading datasets...")
    train_data = datasets.ImageFolder(train_path, transform=train_transform)
    val_data = datasets.ImageFolder(val_path, transform=val_transform)

    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of validation samples: {len(val_data)}")
    print(f"Number of classes: {len(train_data.classes)}")

    # Create data loaders - reduce num_workers for Windows
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=0)

    # Load pre-trained ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_classes = len(train_data.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load existing model if it exists
    model_path = 'pokemon_classifier.pth'
    if os.path.exists(model_path):
        print("Loading existing model...")
        model.load_state_dict(torch.load(model_path))
    
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}%")
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100 * val_correct / val_total
        print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        print("-" * 60)

    print("Training complete!")
    
    # Save with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'pokemon_classifier_{timestamp}.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as '{save_path}'")
    
    # Also save as the default name for the test script
    torch.save(model.state_dict(), 'pokemon_classifier.pth')
    print("Model also saved as 'pokemon_classifier.pth'")

if __name__ == '__main__':
    freeze_support()
    train_model()