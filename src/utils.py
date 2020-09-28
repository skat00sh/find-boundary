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


DEFAULT_STRUCTURE = np.ones((3, 3))


def idx_check(index) -> int:
    return 0 if index < 0 else index


def morph_operation(binary_img, operation, structuring_element=DEFAULT_STRUCTURE):
    binary_img = np.asarray(binary_img)
    structuring_element = np.asarray(structuring_element)
    struct_shape = structuring_element.shape
    morphed_img = np.zeros((binary_img.shape[0], binary_img.shape[1]))
    struct_origin = (
        int((structuring_element.shape[0] - 1) / 2),
        int((structuring_element.shape[1] - 1) / 2),
    )
    for i in range(len(binary_img)):
        for j in range(len(binary_img[0])):
            overlap = binary_img[
                      idx_check(i - struct_origin[0]): i + (struct_shape[0] - struct_origin[0]),
                      idx_check(j - struct_origin[1]): j + (struct_shape[1] - struct_origin[1]),
                      ]
            shape = overlap.shape
            struct_first_row_idx = (int(np.fabs(i - struct_origin[0])) if i - struct_origin[0] < 0 else 0)
            struct_first_col_idx = (int(np.fabs(j - struct_origin[1])) if j - struct_origin[1] < 0 else 0)

            struct_last_row_idx = (struct_shape[0] - 1 - (i + (struct_shape[0] - struct_origin[0]) - binary_img.shape[0])
                                   if i + (struct_shape[0] - struct_origin[0]) > binary_img.shape[0]
                                   else struct_shape[0] - 1
                                   )
            struct_last_col_idx = (struct_shape[1] - 1 - (j + (struct_shape[1] - struct_origin[1]) - binary_img.shape[1])
                                   if j + (struct_shape[1] - struct_origin[1]) > binary_img.shape[1]
                                   else struct_shape[1] - 1
                                   )
            if operation == "erosion":
                if (
                        shape[0] != 0
                        and shape[1] != 0
                        and np.array_equal(np.logical_and(overlap, structuring_element[
                                                                   struct_first_row_idx: struct_last_row_idx + 1,
                                                                   struct_first_col_idx: struct_last_col_idx + 1,
                                                                   ], ),
                                           structuring_element[struct_first_row_idx: struct_last_row_idx + 1,
                                           struct_first_col_idx: struct_last_col_idx + 1, ], )):
                    morphed_img[i, j] = 1
            elif operation == "dilation":
                if (
                        shape[0] != 0
                        and shape[1] != 0
                        and np.logical_and(structuring_element[struct_first_row_idx: struct_last_row_idx + 1,
                                           struct_first_col_idx: struct_last_col_idx + 1, ], overlap, ).any()):
                    morphed_img[i, j] = 1
    return morphed_img



