import numpy as np
import cv2
import open3d as o3d
from PIL import Image
from matplotlib import pyplot
from utils.bmp2tensor import bmp2tensor
from binary_erosion.erosion import erode, erode2
from foreground_extraction.segment import segment2
from edge_detection.edge_detection import edge_detection
from labeling_cc.labeling import labeling
from pc_conversion.pc_conversion import pc_conversion
from icp_evaluator.matchingPointsCloudUseICP import matchingUseIcp


# Data init
# Parts
parts_str = [ 'Part01.ply', 'Part04.ply', 'Part05.ply', \
    'Part07.ply', 'Part11.ply', 'Part12.ply', 'Part14.ply'\
    'Part15.ply', 'Part22.ply', 'Part23.ply', 'Part27.ply'\
    'Part31.ply' ]

# Problems
vol = bmp2tensor('Problem01', [200,200,200])
depth = vol.shape[0]
rows = vol.shape[1]
cols = vol.shape[2]

# Result inits
part_count = [0]*12

# 2-way Canny Edge
pipeline1 = edge_detection(vol.astype('uint8'))

"""
# Kmeans (at the moment only simple thresholding)
pipeline2 = segment2(vol)

# Binary erosion
for num in range(depth):
    pipeline2[num,:,:]=erode(pipeline2[num,:,:],3)

for num in range(vol.shape[1]):
    pipeline2[:,num,:]=erode2(pipeline2[:,num,:],3)
"""

# Labeling
labeled = labeling(vol, pipeline1)

# Point cloud conversion
object_points = pc_conversion(labeled)

for i in range(len(object_points)):
    print("ICP with label " + str(i+1))
    target = object_points[i]
    temp = []
    for j in range(len(parts_str)):
        fname = './SourcePointClouds/'+parts_str[j]
        source = o3d.io.read_point_cloud(fname)
        rmse, transformation = matchingUseIcp(source, target)
        print('\n'+parts_str[j]+' '+str(rmse))
        temp.append(rmse)

    min_rmse = min(temp)

    if min_rmse < 16:
        part_count[temp.index(min_rmse)] = part_count[temp.index(min_rmse)] + 1
    
print(part_count)


