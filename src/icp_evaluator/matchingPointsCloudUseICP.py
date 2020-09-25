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
    source_array =  np.asarray(source.points).transpose()
    target_array =  np.asarray(target.points).transpose()

    cov_s = np.cov(source_array)
    #print(cov_s.shape)
    r_s, _a, _b = np.linalg.svd(cov_s)

    cov_t = np.cov(target_array)
    #print(cov_t.shape)
    r_t, _c, _d = np.linalg.svd(cov_t)

    r_ = r_s.dot(r_t.transpose())
    #print(r_.shape)

    move_t = np.mean(target_array ,axis=1) - np.mean( source_array ,axis=1)

    #print("move_t:",move_t)

    current_transformation = np.identity(4)
    
    #print("3. Colored point cloud registration")

    criteria = o3d.registration.ICPConvergenceCriteria( relative_fitness = 0.0005, #fitnessの変化分がこれより小さくなったら収束
                                                        relative_rmse = 0.0001,      #relative_rmseの変化分がこれより小さくなったら収束 
                                                        max_iteration = 1 )     #反福1回だけする
    est_method = o3d.registration.TransformationEstimationPointToPoint()
    threshold  = 300

    #移動させる
    r_b = np.zeros((4, 4))
    r_b[:3,:3] = r_
    
    r_b[:3,3] = move_t
    r_b[3,3] = 1

    # ######## test
    # r_b[:3,:3] = r_
    # r_b[:3,3] = move_t
    # r_b[0,0] = 1
    # r_b[1,1] = 1
    # r_b[2,2] = 1
    # r_b[3,3] = 1
    # ######## test
    
    #移動&回転
    target.transform(r_b)
    max_iter = [500, 300, 100,80, 60, 40, 30 ,20 ,10, 5 , 4, 3, 2, 1, 0.5 ]

    for i in range(15):

        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius= 0.1, max_nn=30))

        result_icp = o3d.registration.registration_icp(
                                                       source, 
                                                       target, 
                                                       max_iter[i], 
                                                       current_transformation,
                                                       estimation_method = est_method,
                                                       criteria = criteria )
        

        current_rmse = result_icp.inlier_rmse 

        # current_transformation = np.zeros((4,4))

        # # current_transformation[0,0] = 1
        # # current_transformation[1,1] = 1
        # # current_transformation[2,2] = 1
        # # current_transformation[3,3] = 1
        
        
        # current_transformation= copy.deepcopy( result_icp.transformation )

        # current_transformation.flags.writeable = True
        # current_transformation[0,1] = 0
        # current_transformation[0,2] = 0
       
        # current_transformation[1,0] = 0
        # current_transformation[1,2] = 0
        
        # current_transformation[2,0] = 0
        # current_transformation[2,1] = 0
        
        # current_transformation[3,0] = 0
        # current_transformation[3,1] = 0
        # current_transformation[3,2] = 0
        # current_transformation[3,3] = 1

        current_transformation = result_icp.transformation

        
        #o3d.visualization.draw_geometries([target])
        source.transform(current_transformation)
        #o3d.visualization.draw_geometries([target])

        print("iteration {0:02d} fitness {1:.6f} RMSE {2:.6f}".format(i, result_icp.fitness, result_icp.inlier_rmse))

        draw_registration_result(source, target, result_icp.transformation)

    return current_rmse, current_transformation


if __name__ == "__main__":
    # test for two similar parts
    source = o3d.io.read_point_cloud("./src/SourcePointClouds/Part27.ply")
    target = o3d.io.read_point_cloud("./src/SourcePointClouds/Part14.ply")

    #o3d.visualization.draw_geometries([target])

    # trans_ = np.asarray([[-1,0,0,1],   # 4x4 identity matrix，transform matrix
    #                      [0,1,0,1],   # inital the transform matrix
    #                      [0,0,-1,1],   
    #                      [0,0,0,1]])
    # target.transform(trans_)


    matchingUseIcp(source, target)
