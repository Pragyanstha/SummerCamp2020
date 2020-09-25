x = ptCloud.Location(:,1);
y = ptCloud.Location(:,2);
z = ptCloud.Location(:,3);

norm = zeros(size(ptCloud.Location,1), 3);

[Gx, Gy, Gz] = imgradientxyz(vol);

for i=1:size(ptCloud.Location,1)
    a = getGrad(ptCloud.Location(i,:), Gx,Gy,Gz,vol)';
    norm(i,:) = 20*[a(1), a(2), a(3)];
end

quiver3(x,y,z, norm(:,1), norm(:,2), norm(:,3))
