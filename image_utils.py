import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def load_image(path):
    """
    Reads an image from a file path and converts it into a NumPy array.
    """
    # Open the image using PIL
    img = Image.open(path)
    # Convert the image to a NumPy array and return it [cite: 2]
    return np.array(img)

def edge_detection(image):
    """
    Performs edge detection on an image array.
    """
    # Step 1: Convert to grayscale [cite: 6]
    # Check if image has 3 channels (RGB) and average them
    if len(image.shape) == 3:
        # Axis 2 is the color channel axis. Averaging collapses it to 2D.
        gray_image = np.mean(image, axis=2)
    else:
        # If already grayscale, just use it as is
        gray_image = image

    # Step 2: Define the filters (kernels) [cite: 8, 9]
    # KernelY: Detects changes in the vertical direction [cite: 10]
    kernelY = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    # KernelX: Detects changes in the horizontal direction [cite: 10]
    kernelX = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])

    # Step 3: Apply convolution [cite: 12]
    # mode='same' ensures the output size matches the input size
    # boundary='fill', fillvalue=0 ensures zero padding is used [cite: 13]
    edgeY = convolve2d(gray_image, kernelY, mode='same', boundary='fill', fillvalue=0)
    edgeX = convolve2d(gray_image, kernelX, mode='same', boundary='fill', fillvalue=0)

    # Step 4: Compute the magnitude
    # Formula: sqrt(edgeX^2 + edgeY^2) 
    # FIX: Changed *2 to **2 to perform squaring
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG
