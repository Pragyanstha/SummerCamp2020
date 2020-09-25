import numpy as np
import math
import os 
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def bmp2tensor(problem_str, size):
    p = Path('./')
    
    fname = p.joinpath(problem_str)

    voxel = np.zeros(size)
    count = 0
    max_height = size[2]
    height = len(list(fname.glob("*")))
    if (max_height > height):
        skip = 0
    skip = math.ceil(height / max_height)
    idx = 0
    for item in fname.glob('*'):
        if count % skip == 0:

            print (str(item), str(item.absolute()))
            im = Image.open(item.absolute())
            im = im.resize((size[0], size[1]),Image.ANTIALIAS).convert('L')
            im2arr = np.array(im)
            voxel[idx,:,:] = im2arr
            idx = idx + 1

        count = count + 1

    return np.array(voxel)


if __name__ == '__main__':
    # Example
    # Arguments : Folder name, size
    # size being an array of 3 numbers, row, col, height of voxel
    # Return : ndarray with the specified size

    print(bmp2tensor('Problem01', [100,100,100]))
