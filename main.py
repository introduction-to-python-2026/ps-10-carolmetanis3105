import numpy as np
from skimage.filters import median
from skimage.morphology import ball
from PIL import Image

from image_utils import load_image, edge_detection


def main():
    # Load image
    image = load_image("pythonHate.jpg")

    # Noise suppression
    clean = median(image, ball(3))

    # Edge detection
    edges = edge_detection(clean)

    # Threshold
    edge_binary = edges > 50

    # Save result
    result = Image.fromarray((edge_binary * 255).astype(np.uint8))
    result.save("my_edges.png")


if __name__ == "__main__":
    main()
