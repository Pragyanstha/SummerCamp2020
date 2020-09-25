import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot
from utils.bmp2tensor import bmp2tensor
from binary_erosion.erosion import erode, erode2
from foreground_extraction.segment import segment


# Data init
vol = bmp2tensor('Problem01', [200,200,200])
depth = vol.shape[0]
rows = vol.shape[1]
cols = vol.shape[2]


# Kmeans (at the moment only simple thresholding)
vol = segment(vol, 5)


# Binary erosion
for num in range(vol.shape[0]):
    vol[num,:,:]=erode(vol[num,:,:],3)

for num in range(vol.shape[1]):
    vol[:,num,:]=erode2(vol[:,num,:],3)

#Image.fromarray(vol[100,:,:]).show()
pyplot.imshow(vol[10,:,:])
pyplot.show()
"""







Image.fromarray(vol[100,:,:]).show()
"""