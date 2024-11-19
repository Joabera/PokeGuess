import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm  # For progress bar
from PIL import Image
import os
from datetime import datetime

def train_baseline_model(data_folder, num_epochs=20, batch_size=32, learning_rate=0.001):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directories for saving outputs
    misclassified_dir = "misclassifications"
    visualization_dir = "visualizations"
    os.makedirs(misclassified_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    # Basic transformations for Phase 1
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Load datasets
    full_data = datasets.ImageFolder(data_folder, transform=train_transform)
    num_classes = len(full_data.classes)
    print(f"Number of Pokémon classes: {num_classes}")

    # Save class names in order
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

    # Save some visualized transformed images
    for i in range(10):  # Limit to the first 10 images
        img, label = train_data[i]  # Access the raw dataset
        inv_transform = transforms.Compose([
            transforms.Normalize([-0.5 / 0.5] * 3, [1 / 0.5] * 3),
            transforms.ToPILImage(),
        ])
        inv_img = inv_transform(img)
        inv_img.save(os.path.join(visualization_dir, f"train_sample_{i}_class_{class_names[label]}.jpg"))


    # Initialize model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Define loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Training loop
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}...")
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
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

                tepoch.set_postfix(loss=running_loss / total_train, accuracy=100 * correct_train / total_train)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct_train / total_train
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating", unit="batch"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

                # Save misclassified images
                for i in range(len(preds)):
                    if preds[i] != labels[i]:
                        inv_img = transforms.ToPILImage()(inputs[i].cpu())
                        inv_img.save(os.path.join(misclassified_dir, f"misclassified_{class_names[labels[i]]}_as_{class_names[preds[i]]}_{i}.jpg"))

        val_loss /= len(val_loader.dataset)
        val_accuracy = 100 * correct_val / total_val
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'baseline_pokemon_classifier.pth')
            print("Saved Best Model")

        scheduler.step()

    print("Baseline training complete!")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_model_path = f'baseline_pokemon_classifier_{timestamp}.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved as '{final_model_path}'")

if __name__ == '__main__':
    data_folder = 'C:/Users/Joab/Documents/PokeGuess/Pokemons'
    train_baseline_model(data_folder)
