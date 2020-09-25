# coding: utf-8 
import open3d as o3d
import numpy as np
import copy

def draw_registration_result(source, target, transformation):
    """
    visulalization the 2 points cloud
    intpu: source and target as points cloud; 
           transformation as transformation 4x4 matrix
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def matchingUseIcp(source, target):
    """
    mathcing 2 points cloud using ICP
    intpu: source, target as points cloud
    return: rmse and transformation matrix(4x4)
    """
    threshold = 100  # threshold

    trans_init = np.asarray([[1,0,0,0],   # 4x4 identity matrix，transform matrix
                             [0,1,0,0],   # inital the transform matrix
                             [0,0,1,0],   
                             [0,0,0,1]])
    
    # points cloud with ICP
    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())

    rmse = reg_p2p.inlier_rmse
    transformation = reg_p2p.transformation

    return rmse, transformation

if __name__ == "__main__":
    # test for two similar parts
    source = o3d.io.read_point_cloud("./SourcePointClouds/Part14.ply")
    target = o3d.io.read_point_cloud("./SourcePointClouds/Part27.ply")

    # matching 2 points cloud uses ICP
    rmse, transformation = matchingUseIcp(source, target)

    # visualization
    draw_registration_result(source, target, transformation)
    
    # output information
    print("rmse:",rmse) 
    print("transformation: ",transformation) 
