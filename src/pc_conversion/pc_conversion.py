import numpy as np
import open3d as o3d

# Input: labeled voxels
# output: individual point coulds for each object labeled
def pc_conversion(voxel):    
    #get voxel maximum = number of labeled components
    num_objs = np.amax(voxel)
    pcd = o3d.geometry.PointCloud()
    for i in range(num_objs.astype(int)):
        np_points = np.where(voxel == i)
        xyz = np.zeros([len(np_points[0]),3])
        for j in range(len(np_points[0])):
            xyz[j,:] = [np_points[0][j], np_points[1][j], np_points[2][j]]
        #print(xyz)
        pcd.points = o3d.utility.Vector3dVector(xyz)
        o3d.io.write_point_cloud("test.ply", pcd)
        o3d.visualization.draw_geometries([pcd], zoom=0.3412, front=[0.4257, -0.2125, -0.8795], lookat=[2.6172, 2.0475, 1.532], up=[-0.0694, -0.9768, 0.2024])

if __name__ == '__main__':
    test = np.zeros([3,3,3])
    test[0:2,1] = 1
    pc_conversion(test)