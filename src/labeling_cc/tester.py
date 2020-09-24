# coding: utf-8
import numpy as np
import cv2
import os

# test
def create_voxel_sample():
    if os.path.exists("voxel_sample.npy"):
        voxel = np.load("voxel_sample.npy")
        return voxel
    voxel = np.zeros((1441, 1091, 1086), dtype=np.uint8)
    for i in range(1,1441):
        filename = "../../../summercamp_data/Problem01/Problem01_{:04}.bmp".format(i)
        if i % 10 == 0:
            print(filename)
        img = cv2.imread(filename)
        voxel[i,:,:] = img[:,:,0]
    np.save("voxel_sample", voxel)
    return voxel

def create_bool_voxel_sample():
    if os.path.exists("boolean_voxel_sample.npy"):
        voxel = np.load("boolean_voxel_sample.npy")
        return voxel
    voxel = np.zeros((1441, 1091, 1086), dtype=bool)
    for i in range(1,1441):
        filename = "../../../summercamp_data/Problem01/Problem01_{:04}.bmp".format(i)
        if i % 10 == 0:
            print(filename)
        img = cv2.imread(filename)
        edge = cv2.Canny(img,3,12)
        voxel[i,:,:] = (edge != 0)
    np.save("boolean_voxel_sample", voxel)
    return voxel

create_bool_voxel_sample()
