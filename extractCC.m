B = bwconncomp(L_t, 26);
test = zeros(size(L_t,1), size(L_t,2), size(L_t,3), 10);
test_t = zeros(size(L));
c=0;
for i=1:size(B.PixelIdxList,2)

    t = cell2mat(B.PixelIdxList(i));
    if(size(t,1) < 1000)
        continue;
    end
    c = c+1;
    test_t = zeros(size(L));
    test_t(ind2sub(size(L), t)) = 1;
    test(:,:,:,c) = test_t;
    volshow(test_t)
    pause(1)
end