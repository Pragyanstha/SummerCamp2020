import numpy as np

from ..utils.bmp2tensor import bmp2tensor
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

from PIL import Image

voxels=bmp2tensor("Part1",[100,100,100]).astype(np.uint8)


for num in range(voxels.shape[0]):
    voxels1=np.array(cv2.Canny(voxels[num,:,:],3,12))

for num in range(voxels.shape[1]):
    voxels2=(cv2.Canny(voxels[:,num,:],3,12))

output=np.array(voxels1|voxels2)

print(output)

count=np.count_nonzero(output ==255)

print(count)
