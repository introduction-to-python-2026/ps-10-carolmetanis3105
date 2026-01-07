from PIL import Image
import numpy as np
from scipy.signal import convolve2d


def load_image(path):
    img = Image.open(path)

    # Color image → return RGB array
    if img.mode in ("RGB", "RGBA"):
        if img.mode == "RGBA":
            img = img.convert("RGB")
        return np.array(img)

    # Grayscale / edge image → return boolean mask
    gray = np.array(img.convert("L"))
    return gray > 0


def edge_detection(image):
    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image.astype(float)

    kernelY = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ])

    kernelX = np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ])

    edgeX = convolve2d(gray, kernelX, mode="same", boundary="fill", fillvalue=0)
    edgeY = convolve2d(gray, kernelY, mode="same", boundary="fill", fillvalue=0)

    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    return edgeMAG
