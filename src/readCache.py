import numpy as np
from numpy import load
import open3d as o3d
import sys
import re
import matplotlib.pyplot as plt

if (len(sys.argv) == 1):
    fname = 'Problem01/eroded_200.npy'
else:
    fname = sys.argv[1]
    
fname = './cache/' + fname

if(fname.endswith('.ply')):
    pcd = o3d.io.read_point_cloud(fname)
    o3d.visualization.draw_geometries([pcd])

else:
    vol = load(fname)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(vol.astype(np.uint8))
    plt.show()