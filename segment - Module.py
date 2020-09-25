import cv2
import numpy as np
 
        
def image(filename):
        image = cv2.imread(filename)
        label,result = kmeans(image)
        ret, bw_img = cv2.threshold(result,127,255,cv2.THRESH_BINARY)
        return bw_img
    
def kmeans(image):
       #Preprocessing step
       image=cv2.GaussianBlur(image,(7,7),0)
       vectorized=image.reshape(-1,3)
       vectorized=np.float32(vectorized)
       criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
              10, 1.0)
       ret,label,center=cv2.kmeans(vectorized,2,None,
              criteria,10,cv2.KMEANS_RANDOM_CENTERS)
       res = center[label.flatten()]
       segmented_image = res.reshape((image.shape))
       return (label.reshape((image.shape[0],image.shape[1])),
       segmented_image.astype(np.uint8))
              
    
cv2.imshow("Binary Image",image("landscape.jpg"))
cv2.waitKey(0)