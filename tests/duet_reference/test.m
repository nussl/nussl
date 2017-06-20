x = ones(5, 5);

y = zeros(5, 5);

z = x * 15 + (y + 2) * 8;

save('my_matrix_z.mat', z);