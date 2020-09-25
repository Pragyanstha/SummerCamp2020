import open3d as o3d
import numpy as np
from utils.bmp2tensor import bmp2tensor
from pc_conversion.pc_conversion import pc_conversion
from foreground_extraction.segment import segment2

parts_str = [ 'Part01', 'Part04', 'Part05', \
    'Part07', 'Part11', 'Part12', 'Part14', \
    'Part15', 'Part22', 'Part23', 'Part27', \
    'Part31' ]

for i in range(len(parts_str)):
    v = bmp2tensor(parts_str[i], [200, 200, 200])
    v , center = segment2(v)
    center = center.astype('int')
    th = np.sort(center, axis=0)[2][0]
    label = np.where(center == th)[0][0]
    v = v == label
    pcs = pc_conversion(v)
    fname = './DenseParts/' + parts_str[i] + '.ply'
    o3d.io.write_point_cloud(fname, pcs)
    #o3d.visualization.draw_geometries([pcs[0]]) # only one pc

