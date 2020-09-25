import numpy as np
import cv2
              
def segment(image, margin):

    h, bin_edges = np.histogram(image)
    peak_arg = np.argmax(h)
    peak_upper = bin_edges[peak_arg+1]
    segmented_image = image > peak_upper - margin
    return segmented_image

        
def segment2(voxel):

    vectorized=voxel.reshape(-1,1)
    vectorized=np.float32(vectorized)
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10, 1.0)
    ret,label,center=cv2.kmeans(vectorized,3,None,criteria,10,cv2.KMEANS_PP_CENTERS)
    #print("Cluster centers : ")
    #print(center)    
    return label.reshape(voxel.shape), center
    
