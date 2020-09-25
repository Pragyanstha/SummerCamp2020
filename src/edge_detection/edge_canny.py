import numpy as np
import sys
sys.path.append('../')
from utils import bmp2tensor as bmp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

from PIL import Image

voxels=np.array(bmp.bmp2tensor(1,[100,100,100])).astype(np.uint8)


for num in range(voxels.shape[0]):
    voxels1=np.array(cv2.Canny(voxels[num,:,:],100,200))

for num in range(voxels.shape[1]):
    voxels2=(cv2.Canny(voxels[:,num,:],100,200))

np.array(voxels1|voxels2)
