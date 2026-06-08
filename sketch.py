"""Randomized sketch/line-art augmentations.

The original project used a single fixed pencil-sketch transform (grayscale ->
invert -> fixed Gaussian blur -> blend). Real hand-drawn inputs vary a lot, so
this module randomizes the parameters and rotates between several drawn styles
to make the classifier robust to sketches, line art, and low-contrast scans.

Each transform takes a PIL.Image and returns a 3-channel PIL.Image (RGB), so it
slots into a torchvision Compose *before* ToTensor/Normalize.
"""
import random

import numpy as np
from PIL import Image, ImageOps, ImageFilter

try:
    import cv2
    _HAS_CV2 = True
except ImportError:  # cv2 optional; edge/threshold styles fall back to pencil
    _HAS_CV2 = False


def _pencil(img, blur_radius, alpha):
    gray = ImageOps.grayscale(img)
    inv = ImageOps.invert(gray)
    blurred = inv.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    dodged = ImageOps.invert(blurred)
    return Image.blend(gray, dodged, alpha=alpha)  # 'L' mode


def _canny_edges(img, lo, hi):
    arr = np.array(img.convert("L"))
    edges = cv2.Canny(arr, lo, hi)
    edges = 255 - edges  # dark lines on white, like ink on paper
    return Image.fromarray(edges)


def _adaptive_threshold(img, block, c):
    arr = np.array(img.convert("L"))
    th = cv2.adaptiveThreshold(
        arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block, c)
    return Image.fromarray(th)


class RandomizedSketch:
    """Apply a random drawn-style transform with randomized parameters.

    Args:
        p: probability of applying any sketch effect (else returns RGB original,
           so the model still sees some color/photo inputs during fine-tuning).
        styles: which styles to sample from. Available: 'pencil', 'edge',
           'threshold'. 'edge'/'threshold' require opencv; they degrade to
           'pencil' if cv2 is unavailable.
    """

    def __init__(self, p=0.9, styles=("pencil", "edge", "threshold")):
        self.p = p
        if not _HAS_CV2:
            styles = tuple(s for s in styles if s == "pencil") or ("pencil",)
        self.styles = styles

    def __call__(self, img):
        img = img.convert("RGB")
        if random.random() > self.p:
            return img

        style = random.choice(self.styles)
        if style == "pencil":
            out = _pencil(img,
                          blur_radius=random.uniform(8, 28),
                          alpha=random.uniform(0.35, 0.75))
        elif style == "edge":
            lo = random.randint(40, 100)
            out = _canny_edges(img, lo=lo, hi=lo + random.randint(40, 120))
        else:  # threshold
            block = random.choice([7, 9, 11, 15, 21])
            out = _adaptive_threshold(img, block=block, c=random.randint(2, 9))

        # light blur sometimes, to mimic soft pencil / scan
        if random.random() < 0.3:
            out = out.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.5)))

        return out.convert("RGB")  # 1-channel 'L' -> 3-channel RGB
