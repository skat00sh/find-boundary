from pathlib import Path

import numpy as np
from PIL import Image

__all__ = ['get_project_root', 'convert_img_to_binary']


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def convert_img_to_binary(img, threshold=180) -> Image:
    fn = lambda x: 255 if x > threshold else 0
    # Reference to modes used in Image.convert
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    img = img.convert("L").point(fn, mode="1")
    return img



