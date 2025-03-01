import cv2
import numpy as np
from skimage import filters

def analyze_frame(frame):
    """Extracts deepfake detection features from the face image."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Edge detection (to check pixel-level inconsistencies)
    edges = filters.sobel(gray)

    # Compute pixel intensity variance (low variance = possible smoothing artifacts)
    variance = np.var(gray)

    # Compute mean brightness (to detect unnatural lighting)
    brightness = np.mean(gray)

    # Combine all features into an array
    return np.array([variance, brightness, np.mean(edges)])
