% Enhanced NMF (initialization & components estimation)
%   [W,H] = nnmf(V);
%
% Input:
%   V: Matrix to factorize V ~= W*H  [n x m]
%
% Output:
%   W: Final basis matrix [n x r]
%   H: Final weight matrix [r x m]

function [W,H] = nnmf(V)

costf = 'KLD';
varmin = 1e-3;
maxiter = 50;
alpha = 0;                                                      %Smoothness parameter
beta = 0;                                                       %Sparseness parameter
gamma = 0;                                                      %Orthogonality parameter

[W,H] = init_nmf(V);                                            %Initialization
disp('NMF factorization:');
[W,H,E] = nmf(V,W,H,costf,varmin,maxiter,alpha,beta,gamma); %#ok<NASGU>
