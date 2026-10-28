import numpy as np
import pytest
from PIL import Image

from preprocess import HALF_NORM, IMAGENET_NORM, to_tensor


@pytest.mark.parametrize("mode,size", [
    ("RGB", (640, 480)), ("RGBA", (300, 500)), ("P", (100, 100)), ("L", (1000, 1000)),
])
def test_to_tensor_shape_and_dtype(mode, size):
    out = to_tensor(Image.new(mode, size), *IMAGENET_NORM)
    assert out.shape == (1, 3, 224, 224)
    assert out.dtype == np.float32


def test_to_tensor_normalizes():
    # A white image normalized with mean=std=0.5 is exactly 1.0 everywhere.
    out = to_tensor(Image.new("RGB", (300, 300), "white"), *HALF_NORM)
    assert np.allclose(out, 1.0)
