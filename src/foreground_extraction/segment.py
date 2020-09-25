import numpy as np
              
def segment(image, margin):

    h, bin_edges = np.histogram(image)
    peak_arg = np.argmax(h)
    peak_upper = bin_edges[peak_arg+1]
    segmented_image = image > peak_upper - margin
    return segmented_image

        
def segment2(voxel):

        for i in range(voxel.shape[0]):
            label, bw = kmeans(vol[i,:,:])
            vol[i,:,:] = bw
        
        return voxel
    
def kmeans(image):

       vectorized=image.reshape(-1,3)
       vectorized=np.float32(vectorized)
       criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10, 1.0)
       ret,label,center=cv2.kmeans(vectorized,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
       res = center[label.flatten()]
       segmented_image = res.reshape((image.shape))
       return (label.reshape((image.shape[0],image.shape[1])),segmented_image.astype(np.uint8))
              