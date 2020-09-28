import numpy as np
import os
from PIL import Image



PROJECT_ROOT = utils.get_project_root()
DATA_DIR = os.path.join(PROJECT_ROOT, "input")

input_img_path = os.path.join(DATA_DIR, "input.png")
output_img_path = os.path.join(DATA_DIR, "output.png")

input_img = Image.open(input_img_path)
output_img = Image.open(output_img_path)

# Convert image to binary via thresholding
input_img = utils.convert_img_to_binary(input_img, threshold=200)

output_img = utils.convert_img_to_binary(output_img, threshold=200)
