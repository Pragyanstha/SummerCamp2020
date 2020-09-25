import numpy as np
import open3d as o3d
# Input: labeled voxels
# output: individual point coulds for each object labeled
def pc_conversion(voxel):        
    pcd = o3d.geometry.PointCloud()
    np_points = np.where(voxel == 1)
    xyz = np.zeros([len(np_points[0]),3])
    for j in range(len(np_points[0])):
        xyz[j,:] = [np_points[0][j], np_points[1][j], np_points[2][j]]
    print(xyz)
    pcd.points = o3d.utility.Vector3dVector(xyz)
    #o3d.io.write_point_cloud("test.ply", pcd)
    #o3d.visualization.draw_geometries([pcd])

    return pcd

if __name__ == '__main__':
    test = np.zeros([3,3,3])
    test[0:2,1] = 1
    pc_conversion(test)