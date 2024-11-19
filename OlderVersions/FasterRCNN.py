import os
import torch
import xml.etree.ElementTree as ET
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm  # Import tqdm for progress bar

# Paths
images_dir = 'C:/Users/Joab/Documents/PokeGuess/Pokemons'
annotations_dir = 'C:/Users/Joab/Documents/PokeGuess/Annotations'
checkpoint_path = 'pokemon_checkpoint.pth'

# Create a mapping from Pokémon names to unique numeric labels
label_map = {}
label_index = 1  # Start from 1 to leave 0 for background
for label_folder in os.listdir(images_dir):
    if os.path.isdir(os.path.join(images_dir, label_folder)):
        if label_folder not in label_map:
            label_map[label_folder] = label_index
            label_index += 1

# Save the label map to a file for later reference
with open("label_map.txt", "w") as f:
    for name, index in label_map.items():
        f.write(f"{index}: {name}\n")

print("Label map saved to 'label_map.txt'")

# Custom Dataset Class for Pascal VOC-Style Annotations
class PokemonDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None, label_map=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.label_map = label_map
        self.image_files = []
        self.annotation_files = []

        # Load all image and annotation file paths
        for label_folder in os.listdir(images_dir):
            label_path = os.path.join(images_dir, label_folder)
            if os.path.isdir(label_path):
                for image_name in os.listdir(label_path):
                    if image_name.endswith(".jpg") or image_name.endswith(".png"):
                        image_path = os.path.join(label_path, image_name)
                        annotation_path = os.path.join(annotations_dir, label_folder, f"{os.path.splitext(image_name)[0]}.xml")
                        if os.path.exists(annotation_path):
                            self.image_files.append(image_path)
                            self.annotation_files.append(annotation_path)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        annotation_path = self.annotation_files[idx]
        image = Image.open(image_path).convert("RGB")

        # Parse annotation
        boxes = []
        labels = []
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            label_name = obj.find("name").text
            label = self.label_map.get(label_name, 0)  # Use 0 if label not found (background)
            labels.append(label)
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        target = {"boxes": boxes, "labels": labels}
        return image, target

# Define a collate function for the DataLoader
def collate_fn(batch):
    return tuple(zip(*batch))

# Load checkpoint if exists
def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start at the next epoch
        print(f"Resuming training from epoch {start_epoch}")
        return start_epoch
    return 0  # Start from scratch if no checkpoint

# Main training code
if __name__ == "__main__":
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load the dataset
    dataset = PokemonDataset(images_dir=images_dir, annotations_dir=annotations_dir, transform=transform, label_map=label_map)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)

    # Set up the model with your pretrained backbone
    classification_model = models.resnet18()
    classification_model.fc = torch.nn.Linear(classification_model.fc.in_features, 151)
    state_dict = torch.load("fine_tuned_with_pencil_sketch.pth")
    state_dict.pop("fc.weight", None)
    state_dict.pop("fc.bias", None)
    classification_model.load_state_dict(state_dict, strict=False)

    backbone = torch.nn.Sequential(*list(classification_model.children())[:-2])
    backbone.out_channels = 512

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    model = FasterRCNN(
        backbone,
        num_classes=len(label_map) + 1,  # Total Pokémon classes + background
        rpn_anchor_generator=anchor_generator
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Set up optimizer and load checkpoint if available
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    # Training loop with tqdm progress bar
    num_epochs = 5
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            for images, targets in tepoch:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Forward pass
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                # Backward pass and optimize
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                # Update loss for progress bar
                epoch_loss += losses.item()
                tepoch.set_postfix(loss=epoch_loss / len(tepoch))

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

    # Save the final model after all epochs
    torch.save(model.state_dict(), "pokemon_detector_final.pth")
    print("Final model saved as 'pokemon_detector_final.pth'")
