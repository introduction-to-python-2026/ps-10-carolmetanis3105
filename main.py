import numpy as np
from skimage.filters import median
from skimage.morphology import ball
from PIL import Image

from image_utils import load_image, edge_detection


def main():
    # Load image
    image_path = "pythonHate.jpg"   # change if needed
    image = load_image(image_path)

    # Noise suppression
    clean_image = median(image, ball(3))

    # Edge detection
    edgeMAG = edge_detection(clean_image)

    # Thresholding
    threshold = np.mean(edgeMAG)
    edge_binary = edgeMAG > threshold

    # Save result
    edge_image = Image.fromarray((edge_binary * 255).astype(np.uint8))
    edge_image.save("my_edges.png")


if __name__ == "__main__":
    main()

