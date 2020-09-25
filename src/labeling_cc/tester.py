# coding: utf-8
import numpy as np
import cv2
import os
from labeling import *

# test
def create_voxel_sample():
    if os.path.exists("voxel_sample.npy"):
        voxel = np.load("voxel_sample.npy")
        return voxel
    voxel = np.zeros((1441, 1091, 1086), dtype=np.uint8)
    for i in range(1,1441):
        filename = "Problem01/Problem01_{:04}.bmp".format(i)
        if i % 10 == 0:
            print(filename)
        img = cv2.imread(filename)
        voxel[i,:,:] = img[:,:,0]
    np.save("voxel_sample", voxel)
    return voxel

def create_downsampled_voxel_sample():
    if os.path.exists("downsampled_voxel_sample.npy"):
        voxel = np.load("downsampled_voxel_sample.npy")
        return voxel
    voxel = np.zeros((360, 275, 271), dtype=np.uint8)
    for i in range(1,360):
        filename = "Problem01/Problem01_{:04}.bmp".format(i*4)
        if i % 10 == 0:
            print(filename)
        img = cv2.imread(filename)
        img_resize = cv2.resize(img, (271, 275))#notice: (w,h), not(h,w)
        voxel[i,:,:] = img_resize[:,:,0]
    #np.save("downsampled_voxel_sample", voxel)
    return voxel

def create_th_voxel(char_voxel):
    return(char_voxel > 68).astype(np.bool)
