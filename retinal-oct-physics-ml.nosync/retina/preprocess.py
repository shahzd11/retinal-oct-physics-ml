"""Section 2: artifact-aware geometric normalization and spatial micro-slicing."""
import cv2
import numpy as np

W_TARGET, H_TARGET = 750, 500
GRID = 50                      # 50x50 px micro-grids
N_COLS, N_ROWS = W_TARGET // GRID, H_TARGET // GRID   # 15 x 10 = 150 grids
WHITE_THRESH = 240             # I_white >= 240 treated as acquisition artifact


def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"cannot read {path}")
    return img


def normalize(img):
    """Sec 2.1: white-wedge mask -> tight bbox crop -> symmetric crop to 3:2 ->
    cubic resize to 750x500 -> residual white pixels padded with black."""
    # 1. binary mask of biological data (not near-white) and its bounding box
    mask = img < WHITE_THRESH
    ys, xs = np.nonzero(mask)
    if len(ys):
        img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # 2. symmetric crop to R_target = W/H
    h, w = img.shape
    r_target, r_cur = W_TARGET / H_TARGET, w / h
    if r_cur > r_target:            # too wide -> crop horizontally
        new_w = int(round(h * r_target))
        x0 = (w - new_w) // 2
        img = img[:, x0:x0 + new_w]
    elif r_cur < r_target:          # too tall -> crop vertically
        new_h = int(round(w / r_target))
        y0 = (h - new_h) // 2
        img = img[y0:y0 + new_h, :]

    # 3. cubic resize to 750x500
    img = cv2.resize(img, (W_TARGET, H_TARGET), interpolation=cv2.INTER_CUBIC)

    # 4. pad residual high-intensity boundary artifacts with absolute black
    img = img.copy()
    img[img >= WHITE_THRESH] = 0
    return img


def micro_slice(img):
    """Sec 2.2: split 500x750 image into (150, 50, 50) grids.
    Grid index g = row * 15 + col (row-major; matches Fig. 7)."""
    return (img.reshape(N_ROWS, GRID, N_COLS, GRID)
               .transpose(0, 2, 1, 3)
               .reshape(N_ROWS * N_COLS, GRID, GRID))
