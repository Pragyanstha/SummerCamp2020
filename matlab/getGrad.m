function g = getGrad(coord, Gx,Gy,Gz, vol)
    r = coord(1)+1;
    c = coord(2)+1;
    s = size(vol,3)-coord(3);
    a = [Gx(r,c,s); Gy(r,c,s); Gz(r,c,s)];
    a = [a(1)-1; a(2)-1;-a(3)];
    g = a/norm(a);
end