import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image, ImageOps, ImageFilter
import os
from datetime import datetime

# Custom pencil sketch transformation
class PencilSketchTransform:
    def __call__(self, img):
        # Ensure the image is in RGB mode to handle transparency issues
        img = img.convert("RGB")

        # Convert to grayscale
        gray_image = ImageOps.grayscale(img)

        # Invert colors
        inverted_image = ImageOps.invert(gray_image)

        # Apply Gaussian blur
        blurred = inverted_image.filter(ImageFilter.GaussianBlur(radius=21))

        # Invert the blurred image
        inverted_blurred = ImageOps.invert(blurred)

        # Blend the original grayscale with the inverted blurred image
        pencil_sketch = Image.blend(gray_image, inverted_blurred, alpha=0.5)

        # Convert to tensor and normalize
        pencil_sketch = transforms.ToTensor()(pencil_sketch)
        pencil_sketch = transforms.Normalize([0.5], [0.5])(pencil_sketch)
        
        # Repeat grayscale to make it 3 channels
        pencil_sketch = pencil_sketch.repeat(3, 1, 1)
        
        return pencil_sketch

def save_transformed_images(loader, folder_name="transformed_images", num_images=5):
    os.makedirs(folder_name, exist_ok=True)
    for i, (img, _) in enumerate(loader):
        if i >= num_images:
            break
        # Save transformed images
        img = img[0]  # Take the first image in the batch
        img = (img * 0.5) + 0.5  # Unnormalize for viewing
        img = transforms.ToPILImage()(img)
        img.save(f"{folder_name}/image_{i+1}.jpg")

def fine_tune_with_pencil_sketch(data_folder, num_epochs=10, batch_size=32, learning_rate=0.001):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Enhanced transformations with Pencil Sketch effect
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        PencilSketchTransform(),  # Apply custom pencil sketch effect
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        PencilSketchTransform(),  # Apply custom pencil sketch effect for validation
    ])

    # Load datasets
    full_data = datasets.ImageFolder(data_folder, transform=train_transform)
    train_size = int(0.8 * len(full_data))
    val_size = len(full_data) - train_size
    train_data, val_data = torch.utils.data.random_split(full_data, [train_size, val_size])

    val_data.dataset.transform = val_transform
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Save a few transformed images from the training set to preview
    save_transformed_images(train_loader)

    # Initialize model and load baseline checkpoint
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(full_data.classes))
    model.load_state_dict(torch.load('baseline_pokemon_classifier.pth', map_location=device))
    model = model.to(device)

    # Define loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # Fine-tuning loop
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}...")
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
            torch.save(model.state_dict(), 'fine_tuned_with_pencil_sketch.pth')
            print("Saved Best Model")

        scheduler.step()

    print("Fine-tuning with pencil sketch effect complete!")

if __name__ == '__main__':
    data_folder = 'C:/Users/Joab/Documents/PokeGuess/Pokemons'
    fine_tune_with_pencil_sketch(data_folder)
