function writeImg(fname, voxel)
pname = strcat(fname,'/',fname,'_');
extname = '.bmp';
S = size(voxel,3);
for i=1:S
    nname = sprintf('%04d', i);
    fname = strcat(pname, nname, extname);
    imwrite(voxel(:,:,i), fname);
end