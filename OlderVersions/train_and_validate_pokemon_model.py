import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import os
from datetime import datetime

# Custom transformation to enhance outlines but keep colors
class CustomTransform:
    def __call__(self, img):
        # Randomly apply edge enhancement to mimic an outline effect
        if torch.rand(1).item() > 0.5:
            img = F.autocontrast(img)
        
        # Randomly invert colors to simulate variety
        if torch.rand(1).item() > 0.5:
            img = F.invert(img)
        
        # Convert to tensor and normalize
        img = F.to_tensor(img)
        img = F.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        
        return img

def train_and_validate(data_folder, num_epochs=10, batch_size=32, learning_rate=0.001, checkpoint_path=None, freeze_layers=False):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        CustomTransform(),  # Only custom transforms without grayscale
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        CustomTransform(),  # Only custom transforms without grayscale
    ])

    # Load datasets
    full_data = datasets.ImageFolder(data_folder, transform=train_transform)
    num_classes = len(full_data.classes)
    print(f"Number of Pokémon classes: {num_classes}")

    # Save class names in training order with UTF-8 encoding
    class_names = full_data.classes
    with open("class_names.txt", "w", encoding="utf-8") as f:
        for class_name in class_names:
            f.write(class_name + "\n")
    
    train_size = int(0.8 * len(full_data))
    val_size = len(full_data) - train_size
    train_data, val_data = torch.utils.data.random_split(full_data, [train_size, val_size])

    val_data.dataset.transform = val_transform
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load checkpoint for fine-tuning
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))

    model = model.to(device)

    # Freeze layers if specified (useful for fine-tuning)
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    # Define loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Training loop
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct_train / total_train
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_accuracy = 100 * correct_val / total_val
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'fine_tuned_pokemon_classifier.pth')
            print("Saved Best Model")

        scheduler.step()

    print("Fine-tuning complete!")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_model_path = f'fine_tuned_pokemon_classifier_{timestamp}.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved as '{final_model_path}'")

if __name__ == '__main__':
    data_folder = 'C:/Users/Joab/Documents/PokeGuess/Pokemons'
    train_and_validate(data_folder, checkpoint_path='fine_tuned_pokemon_classifier.pth', freeze_layers=False)
