import numpy as np
import sys
sys.path.append('../')
from utils import bmp2tensor_parts as bmp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from PIL import Image

voxels=np.array(bmp.bmp2tensor(1,[100,100,100]))

print(voxels)
