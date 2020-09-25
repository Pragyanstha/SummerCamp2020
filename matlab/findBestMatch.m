function [sptc, rmse] = findBestMatch(inptc)
plys = dir('./SourcePointClouds');
rmses = zeros(size(plys,1),1);
for i=3:size(plys,1)
    fname = plys(i).name;
    ptCloud = pcread(strcat('./SourcePointClouds/',fname));
    [~, ~, rmses(i)] = pcregistericp(inptc, ptCloud);
end

% Just for visualization
[rmse, idx] = min(rmses(3:end));
fname = plys(idx+2).name
sptc = pcread(strcat('./SourcePointClouds/',fname));
