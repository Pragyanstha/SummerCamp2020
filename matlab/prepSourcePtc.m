function part_ptc = prepSourcePtc(fname, skip, downsample)
part = loadVolume(fname, skip, downsample);
part = imsegkmeans3(part, 2);
part = part == 2;
part_ptc = convert2pc(part);
end