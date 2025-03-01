import cv2
import numpy as np
from skimage import filters

def analyze_frame(frame):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = filters.sobel(gray)
    variance = np.var(gray)
    brightness = np.mean(gray)
    return np.array([variance, brightness, np.mean(edges)])
