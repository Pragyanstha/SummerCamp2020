import numpy as np
import cv2
import open3d as o3d
import sys
from numpy import load, save
from os import path
from pathlib import Path
import os
from PIL import Image
from matplotlib import pyplot
from utils.bmp2tensor import bmp2tensor
from binary_erosion.erosion import erode
from foreground_extraction.segment import segment2
from edge_detection.edge_detection import edge_detection
from labeling_cc.labeling import labeling_module
from pc_conversion.pc_conversion import pc_conversion
from icp_evaluator.matchingPointsCloudUseICP import matchingUseIcp

def getCachePath(num_step):
    return cache_folder + PROBLEM_STR +'/'+caching_steps[num_step]+'_'+str(VOXEL_SIZE)+EXT

def getCCVPath(num):
    return cache_folder + PROBLEM_STR + '/Components/' + 'cc_'+ str(VOXEL_SIZE)+'_'+str(num)+'.ply'

if (len(sys.argv) != 3):
    print('Running with Probelm01 and size 200')
    PROBLEM_STR = 'Problem01'
    VOXEL_SIZE = '200'
else:     
    commandlines = sys.argv
    #print(commandlines)
    
    PROBLEM_STR = commandlines[1]
    VOXEL_SIZE = int(commandlines[2])

EXT = '.npy'

# Data init
# Parts
parts_str = [ 'Part01.ply', 'Part04.ply', 'Part05.ply', \
    'Part07.ply', 'Part11.ply', 'Part12.ply', 'Part14.ply',\
    'Part15.ply', 'Part22.ply', 'Part23.ply', 'Part27.ply',\
    'Part31.ply' ]

# Result inits
part_count = [0]*12

# Cache files init
cache_folder = './cache/'
caching_steps = ['vol', 'canny_2way', 'segmented', 'eroded','labeled']

# Problems
fname = getCachePath(0)
if(path.exists(fname)):
    vol = load(fname)
else:
    vol = bmp2tensor(PROBLEM_STR, [VOXEL_SIZE,VOXEL_SIZE,VOXEL_SIZE])
    save(fname, vol)

depth = vol.shape[0]
rows = vol.shape[1]
cols = vol.shape[2]


# 2-way Canny Edge
fname = getCachePath(1)
if(path.exists(fname)):
    pipeline1 = load(fname)
else:
    pipeline1 = edge_detection(vol.astype('uint8'))
    save(fname, pipeline1)


# Kmeans
fname = getCachePath(2)
if(path.exists(fname)):
    pipeline2 = load(fname)
else:
    pipeline2, center = segment2(vol)
    center = center.astype('int')
    th = np.sort(center, axis=0)[2][0]
    label = np.where(center == th)[0][0]
    pipeline2 = pipeline2 == label  # extracts only the labels 
    save(fname, pipeline2)

# Binary erosion
fname = getCachePath(3)
if(path.exists(fname)):
    pipeline2 = load(fname)
else:
    for num in range(depth):
        pipeline2[num,:,:]=erode(pipeline2[num,:,:],3,1)

    for num in range(vol.shape[1]):
        pipeline2[:,num,:]=erode(pipeline2[:,num,:],3,1)
    
    save(fname, pipeline2)


# Labeling
fname = getCachePath(4)
if(path.exists(fname)):
    labeled = load(fname)
else:
    labeled = labeling_module(vol, pipeline1, pipeline2)
    save(fname, labeled)


# Point cloud conversion
pc_base = cache_folder + PROBLEM_STR + '/Components'
if(not path.exists(pc_base)):
    os.mkdir(pc_base)
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
fname = getCCVPath(1)
if(path.exists(fname)):
    object_points = []
    for item in Path(pc_base).glob("*"):
        object_points.append(o3d.io.read_point_cloud(str(item)))
else:
    #get voxel maximum = number of labeled components
    num_objs = np.amax(labeled.astype('int'))
    object_points = []
    for i in range(1,num_objs.astype(int)+1):
        voxel = labeled == i
        pcd = pc_conversion(voxel)
        o3d.io.write_point_cloud(getCCVPath(i),pcd)
        object_points.append(pcd)


print('Number of Components inside the bucket : ' + str(len(object_points)))

for i in range(len(object_points)):
    print("ICP with label " + str(i+1)+ '\n')
    source = object_points[i]
    #o3d.visualization.draw_geometries([source])
    temp = []
    for j in range(len(parts_str)):
        fname = './DenseParts/'+parts_str[j]
        target = o3d.io.read_point_cloud(fname)
        #o3d.visualization.draw_geometries([target])
        rmse, transformation = matchingUseIcp(source, target)
        print(parts_str[j]+' '+str(rmse))
        if (rmse <= 1e-8):
            temp.append(float('inf'))
        else:
            temp.append(rmse)

    min_rmse = min(temp)

    if min_rmse < 16:
        part_count[temp.index(min_rmse)] = part_count[temp.index(min_rmse)] + 1
    
print(part_count)


