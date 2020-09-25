import numpy as np
import cv2

def erode(src, ksize, it):
    kernel = np.ones((ksize, ksize), np.uint8) 
    dst = cv2.erode(src.astype(np.uint8),kernel, iterations=it)
    return dst
