import numpy as np

from ..utils.bmp2tensor import bmp2tensor
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from PIL import Image

voxels=bmp2tensor("Part1",[100,100,100])

print(voxels)

voxels_2=np.ravel(voxels)

print(voxels_2.size)

ave=np.mean(voxels_2)

print(ave)
print(np.max(voxels_2))
print(np.min(voxels_2))
