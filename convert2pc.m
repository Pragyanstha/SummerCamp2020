function ptc = convert2pc(v)
ind = find(v);
[row, col, slice] = ind2sub(size(v), ind);
coords = [row-1 col-1 size(v,3)-slice];
ptc = pointCloud(coords);

