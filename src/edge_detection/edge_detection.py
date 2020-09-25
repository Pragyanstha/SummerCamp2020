import numpy as np
import cv2
from PIL import Image

def edge_detection(voxel):
    voxels1 = np.zeros(voxel.shape)
    voxels2 = np.zeros(voxel.shape)
    
    for num in range(voxel.shape[0]):
        
        voxels1[num,:,:]=cv2.Canny(voxel[num,:,:],3,12)

    for num in range(voxel.shape[1]):
        voxels2[:,num,:]=cv2.Canny(voxel[:,num,:],3,12)

    return np.array(np.logical_or(voxels1, voxels2))
