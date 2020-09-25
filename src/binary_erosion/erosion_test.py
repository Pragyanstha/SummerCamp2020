import numpy as np
import cv2
from scipy import ndimage
import skimage
from skimage import morphology
from skimage.morphology import reconstruction
from PIL import Image

#in=np.ndarray
#test=np.arange(48).reshape(3,2,8)
def erode(src, ksize=3):
    # 入力画像のサイズを取得
    h, w = src.shape
    # 入力画像をコピーして出力画像用配列を生成
    dst = src.copy()
    # 注目領域の幅
    d = int(1)

    for y in range(0, h):
        for x in range(0, w):
            roi = src[y-d:y+d+1, x-d:x+d+1]
            if roi.size - np.count_nonzero(roi)>0:
                dst[y][x] = 0

    return dst

def erode2(src, ksize=3):
    # 入力画像のサイズを取得
    h, w = src.shape
    # 入力画像をコピーして出力画像用配列を生成
    dst = src.copy()
    # 注目領域の幅
    d = int(1)

    for y in range(0, h):
        for x in range(0, w):
            roi = src[y, x-d:x+d+1]
            if roi.size - np.count_nonzero(roi)>0:
                dst[y][x] = 0

    return dst

test=np.array(Image.open('aaa.jpg').convert('L'))
#struct = ndimage.generate_binary_structure(3, 3)

"""
test=np.array([[[ 0, 1, 1, 1, 1, 1, 1, 1],
                [ 1, 1, 1, 0, 0, 1, 1, 1],
                [ 1, 1, 1, 0, 0, 1, 1, 1]],

                [[1, 1, 1, 1, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1, 0, 1],
                 [0, 0, 1, 1, 1, 1, 0, 1]],

                [[1, 1, 1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1]],

                 [[1, 1, 1, 1, 1, 1, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1]]])
"""

print(test)
print("¥n")
#test=np.vsplit(test)
#print(test[0])
#kernel = np.ones((3, 3), np.uint8)

"""
for num in range(test.shape[0]):
    print(num)
    test[num,:]=erode(test[num,:],3)
"""

test[:,:]=erode2(test[:,:],3)
    #morphology.binary_erosion(test[num], struct).astype(np.uint8)

#print(test)

#for num in range(test.shape[1]):
#    print(num)
#    test[:,num,:]=erode2(test[:,num,:],3)
    #morphology.binary_erosion(test[num], struct).astype(np.uint8)
pil_img = Image.fromarray(test)
print(pil_img.mode)

pil_img.save('erosion2.jpg')
#print(test)
#a=np.linspace(1)
#testout = reconstruction(test,a, method='erosion').astype(np.uint8)


#testout=morphology.binary_erosion(test, morphology.diamond(1))
#astype(np.uint8)

#out=morphology.binary_erosion(in, morphology.diamond(1)).astype(np.uint8)
#print(out)
#print(testout)
