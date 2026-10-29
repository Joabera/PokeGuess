"""PIL image -> model input, matching the torchvision transform used in training:
Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(mean, std)."""
import numpy as np
from PIL import Image

# (mean, std) presets. ConvNeXt trained with ImageNet stats, legacy ResNet-18 with 0.5.
IMAGENET_NORM = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HALF_NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def to_tensor(image, mean, std):
    """Return a (1, 3, 224, 224) float32 array."""
    img = image.convert("RGB")
    w, h = img.size
    if w <= h:
        nw, nh = 256, int(256 * h / w)
    else:
        nw, nh = int(256 * w / h), 256
    img = img.resize((nw, nh), Image.BILINEAR)
    left, top = int(round((nw - 224) / 2.0)), int(round((nh - 224) / 2.0))
    img = img.crop((left, top, left + 224, top + 224))
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - np.array(mean, dtype=np.float32)) \
        / np.array(std, dtype=np.float32)
    return arr.transpose(2, 0, 1)[None].astype(np.float32)
