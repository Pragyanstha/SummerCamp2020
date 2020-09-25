folders = dir('.');
for i=1:size(folders, 1)
    fname = folders(i).name;
    if (fname(1) == 'P' && fname(2) == 'a')
        ptClouds = prepSourcePtc(fname, 1, 0.5);
        pcwrite(ptClouds, fname)
    end
end