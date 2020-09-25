# coding: utf-8
import numpy as np
import cv2

from union_find import UnionFind

def create_bool_voxel(char_voxel):

    #if os.path.exists("boolean_voxel_sample.npy"):
    #    voxel = np.load("boolean_voxel_sample.npy")
    #    return voxel
    voxel = np.zeros(char_voxel.shape, dtype=bool)
    for i in range(1,char_voxel.shape[0]):
        #filename = "../../../summercamp_data/Problem01/Problem01_{:04}.bmp".format(i)
        if i % 10 == 0:
            print(i)
        img = char_voxel[i,:,:]#cv2.imread(filename)
        edge = cv2.Canny(img,3,12)
        voxel[i,:,:] = (edge != 0)
    voxel[0,:,:]=False
    voxel[:,0,:]=False
    voxel[:,:,0]=False
    voxel[char_voxel.shape[0]-1,:,:]=False
    voxel[:,char_voxel.shape[1]-1,:]=False
    voxel[:,:,char_voxel.shape[2]-1]=False
    np.save("boolean_voxel_sample", voxel)
    return voxel

def get_normals(char_voxel, targets):
    target_length = targets.shape[0]
    wei = [1,2,1,2,4,2,1,2,1]
    p1 = [-1,0,1,-1,0,1,-1,0,1]
    p2 = [-1,-1,-1,0,0,0,1,1,1]
    sobel = np.zeros((target_length, 3), np.float32)
    tx = targets[:,0]
    ty = targets[:,1]
    tz = targets[:,2]

    for i in range(9):
        sobel[:, 0] += (char_voxel[tx+1, ty+p1[i], tz+p2[i]] - char_voxel[tx-1, ty+p1[i], tz+p2[i]]) * wei[i]
        sobel[:, 1] += (char_voxel[tx+p1[i], ty+1, tz+p2[i]] - char_voxel[tx+p1[i], ty-1, tz+p2[i]]) * wei[i]
        sobel[:, 2] += (char_voxel[tx+p2[i], ty+p1[i], tz+1] - char_voxel[tx+p2[i], ty+p1[i], tz-1]) * wei[i]
    length = np.sqrt(sobel[:,0]*sobel[:,0] + sobel[:,1]*sobel[:,1] + sobel[:,2]*sobel[:,2])
    length[length<=0.00001]=1.0
    normals = np.zeros((target_length, 3), np.float32)
    normals[:,0] = sobel[:,0]/length
    normals[:,1] = sobel[:,1]/length
    normals[:,2] = sobel[:,2]/length
    return normals
def get_normal(char_voxel, x,y,z):
    if x<2 or y<2 or z<2 or x > char_voxel.shape[0]-3 or y > char_voxel.shape[1]-3 or z > char_voxel.shape[2]-3:
        return np.array([0.5,0.5,0.5])
    gaus = [[1,2,1],[2,4,2],[1,2,1]]
    dx = np.average(char_voxel[x+1,y-1:y+2,z-1:z+2],weights = gaus) - np.average(char_voxel[x-1,y-1:y+2,z-1:z+2],weights = gaus)
    dy = np.average(char_voxel[x-1:x+2,y+1,z-1:z+2],weights = gaus) - np.average(char_voxel[x-1:x+2,y-1,z-1:z+2],weights = gaus)
    dz = np.average(char_voxel[x-1:x+2,y-1:y+2,z+1],weights = gaus) - np.average(char_voxel[x-1:x+2,y-1:y+2,z-1],weights = gaus)
    length = np.sqrt(dx*dx+dy*dy+dz*dz)
    if length<=0.000000001:
        return np.array([0.5,0.5,0.5])
    return np.array([0.5 + 0.5*dx/length,0.5 + 0.5*dy/length,0.5 + 0.5*dz/length])

def labeling(char_voxel, bool_voxel):
    # labeling target is boundary voxels
    nonzero = np.nonzero(bool_voxel)
    targets = np.array(nonzero).transpose()
    normals = get_normals(char_voxel, targets)
    idx = np.zeros(char_voxel.shape, dtype=np.int32)
    labels = np.full(char_voxel.shape, -1, dtype=np.int32)
    target_length = targets.shape[0]
    uf = UnionFind(target_length)
    # connection kernel(3x3)
    p1 = [-1,-1,-1,0,0,0,1,1,1]
    p2 = [-1,0,1,-1,0,1,-1,0,1]
    # connection kernel(2x2)
    q1 = [-1,-1,-1,0,0]
    q2 = [-1,0, 1,-1,0]

    arr = np.arange(target_length)
    print(idx[nonzero].shape)
    idx[nonzero] = arr
    for id, p in enumerate(targets):
        if id % 10000 == 0:
            print(id)
        #idx[p[0], p[1], p[2]] = id
        for i in range(9):
            if labels[p[0], p[1]+p1[i], p[2]+p2[i]]==-1:
                # not in boundary voxel
                continue
            idn = idx[p[0], p[1]+p1[i], p[2]+p2[i]]
            #if 0.1 < np.dot(normals[id,:], normals[idn, :]):
            #    uf.union(idn, id)
            uf.union(idn, id)
        #if p[0]<2 or p[1]<2 or p[2] < 2:

        #    labels[p[0], p[1], p[2]] = uf.find(id)
        #    continue
        for i in range(9):
            if labels[p[0]-1, p[1]+p1[i], p[2]+p2[i]]==-1:
                # not in boundary voxel
                continue
            idn = idx[p[0]-1, p[1]+p1[i], p[2]+p2[i]]
            #if 0.1 < np.dot(normals[id,:], normals[idn, :]):
            #    uf.union(idn, id)
            uf.union(idn, id)
        labels[p[0], p[1], p[2]] = uf.find(id)
    # labels lookup table
    lut = np.full(target_length, -1, dtype=np.int32)
    lbl_max = 0
    for i in range(target_length):
        fi = uf.find(i)
        if fi == i and uf.count[i]>10:
            lut[i]=lbl_max
            lbl_max += 1
    print("label count: {}".format(lbl_max))
    for i in range(target_length):
        fi = uf.find(i)
        if lut[fi]!=-1:
            labels[targets[i,0], targets[i,1], targets[i,2]] = lut[fi]
    return labels
