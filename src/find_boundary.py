import numpy as np
import os
from PIL import Image
import utils
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.ndimage.measurements import label

import matplotlib.pyplot as plt

PROJECT_ROOT = utils.get_project_root()
DATA_DIR = os.path.join(PROJECT_ROOT, "input")

input_img_path = os.path.join(DATA_DIR, "input.png")
output_img_path = os.path.join(DATA_DIR, "output.png")

input_img = Image.open(input_img_path)
output_img = Image.open(output_img_path)

# Convert image to binary via thresholding
input_img = utils.convert_img_to_binary(input_img, threshold=200)

output_img = utils.convert_img_to_binary(output_img, threshold=200)

## Explore existing options to do the task
# kernel = np.ones((3, 3), np.uint8) #To allow every type of connection
#
# eroded_img = binary_erosion(input_img, structure=kernel).astype(int)
# plt.imshow(eroded_img,cmap='Greys',  interpolation='nearest')
#
# dilated_img = binary_dilation(input_img, structure=kernel).astype(int)
# plt.imshow(dilated_img,cmap='Greys',  interpolation='nearest')
#
# diff = dilated_img - eroded_img
# plt.imshow(diff,cmap='Greys',  interpolation='nearest')
# plt.show()

# Assignment 1 : Find surface borders Using numpy and Python only
kernel = np.ones((3, 3), np.uint8)
ero = utils.morph_operation(input_img, "erosion", kernel).astype(int)
# plt.imshow(ero, cmap='Greys', interpolation='nearest')
# plt.show()
dil = utils.morph_operation(input_img, "dilation", kernel).astype(int)
# plt.imshow(dil, cmap='Greys', interpolation='nearest')
# plt.show()
diff = dil - ero
plt.imshow(diff, cmap='Greys', interpolation='nearest')
plt.show()

# Assignment 2 : List of all connected components in the image with boundary
labeled,comp = label(diff,kernel) #label all connected components from 1,2,....n

h,w = diff.shape
indices = np.indices((w,h)).T[:,:,[0,1]] #create a array of indices of a grid
buildings={}
for i in range(1,comp+1):
    # print(indices[labeled==i])
    buildings[i] = indices[labeled == i]
