
L = imsegkmeans3(single(vol), 2);
L = edge3(erodedBW, 'approxcanny', 0.1);
ind = find(L);
[row, col, slice] = ind2sub(size(erodedBW), ind);
coords = [row-1 col-1 size(erodedBW,3)-slice];
ptCloud = pointCloud(coords);
pcshow(ptCloud)
