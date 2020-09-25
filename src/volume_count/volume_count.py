import numpy as np
#import sys

#sys.path.append('../')
from ..utils.bmp2tensor_parts import bmp2tensor
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from PIL import Image

voxels=bmp2tensor("Part4",[100,100,100])

print(voxels)

voxels_2=np.ravel(voxels)

print(voxels_2.size)
count=np.count_nonzero(voxels_2 >100 )


print(count)
