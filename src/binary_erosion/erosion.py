import numpy as np

def erode(src, ksize=3):
    h, w = src.shape
    dst = src.copy()
    d = int(1)

    for y in range(0, h):
        for x in range(0, w):
            roi = src[y-d:y+d+1, x-d:x+d+1]
            if roi.size - np.count_nonzero(roi)>0:
                dst[y][x] = 0

    return dst

def erode2(src, ksize=3):
    h, w = src.shape
    dst = src.copy()
    d = int(1)

    for y in range(0, h):
        for x in range(0, w):
            roi = src[y, x-d:x+d+1]
            if roi.size - np.count_nonzero(roi)>0:
                dst[y][x] = 0

    return dst