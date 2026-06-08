"""Shared inference logic for the PokeGuess classifier.

Used by both the Tkinter GUI (test_pokemon_model_gui.py) and the FastAPI
web app (app.py) so the model definition / preprocessing lives in one place.
"""
import torch
import torch.nn as nn
from torchvision import transforms, models


def load_class_names(path="best_sketch_classes.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _build_backbone(backbone, num_classes):
    if backbone == "convnext_tiny":
        m = models.convnext_tiny()
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif backbone == "resnet18":
        m = models.resnet18()
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"unknown backbone {backbone}")
    return m


# Normalization presets. The new ConvNeXt model trained with ImageNet stats;
# the legacy ResNet-18 trained with [0.5,0.5,0.5].
IMAGENET_NORM = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
HALF_NORM = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


class PokemonClassifier:
    def __init__(self, model_path, class_names, backbone="convnext_tiny",
                 norm=IMAGENET_NORM):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names

        self.model = _build_backbone(backbone, len(class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        # Normalization must match what the checkpoint was trained with.
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(*norm)
        ])

    def predict(self, image, top_k=1):
        """Return a list of (class_name, confidence_pct) for the top_k classes."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        img = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            top_probs, top_idxs = torch.topk(probs, min(top_k, len(self.class_names)))

        return [
            (self.class_names[idx.item()], prob.item() * 100)
            for prob, idx in zip(top_probs, top_idxs)
        ]
