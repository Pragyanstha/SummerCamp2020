ptCloudA = pcdownsample(ptCloud, 'gridAverage', 2);
[labels, numClusters] = pcsegdist(ptCloudA, 3);

for i=1:numClusters
    
    pts_idx = find(labels == i);
    pts = select(ptCloudA, pts_idx);
    pcshow(pts)
    pause(1)
end