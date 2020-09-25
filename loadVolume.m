function vol = loadVolume(fname, skip, downsample)

S = size(dir(fname),1) - 2; 
pname = strcat(fname, '/',fname,'_');
extname = '.bmp';

for i = 1:S
    if mod(i,skip+1)~= 0
        continue;
    end
    nname = sprintf('%04d', i);
    fname = strcat(pname, nname, extname);
    vol(:,:,floor(i/(skip+1))+mod(i,skip+1)) = rgb2gray(imresize(imread(fname), downsample));
end