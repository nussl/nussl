% Initial components estimation for the NMF
%   [W,H] = init_nmf(V);
%
% Input:
%   V: matrix to factorize V ~= W*H  [n x m]
%
% Output:
%   W: initial basis matrix [n x r]
%   H: initial weight matrix [r x m]

function [W,H] = init_nmf(V)

% Parameters:

R = 20;                                                         %20 components maximum
varmin = 1e-2;                                                  %1e-2 of minimal variation
maxiter = 20;                                                   %20 iterations maximum
costf = 'CKL';                                                  %CKL NMF
alpha = 3e2;                                                    %Temporal continuity parameter (unnormalized!) (3e2)
beta = 4e3;                                                     %Sparseness parameter (unnormalized!) (4e4)
gamma = 0;                                                      %Orthogonality parameter
T = 1;                                                          %Treshold = 1% of the energy

% NNDSVD initialization + first coarse NMF:

disp('Initial components estimation:');
[W,H] = nndsvd(V,R);
[W,H,E] = nmf(V,W,H,costf,varmin,maxiter,alpha,beta,gamma); %#ok<NASGU>

% Estimation of the relevant components:

eps = 2.2204e-16;
H1 = H./(ones(R,1)*sum(H,1)+eps);                               %Frame normalization (increases the energy of the lone low level parts)
M = 100*mean(H1,2)';                                            %Mean energy of each component (%)
f = find(M>=T);                                                 %Indexes vector of the relevant components
r = length(f);                                                  %Number of relevant components
W = W(:,f);                                                     %Relevant components in W
H = H(f,:);                                                     %Relevant components in H
disp([num2str(r),' initial components estimated']); disp('');

% Increase of the low level values:

o = ones(1,r);
s = sum(V(:));
v = sum(V,2)*r/s;                                     %Normalized upgrade filter for W (sum(W(:))=r)
W = W+1e-3*v*o;                                       %Upgrade of the low-level values in W
v = sum(V,1)*sum(H(:))/s;                             %Normalized upgrade filter for H
H = H+1e-3*o'*v;                                      %Upgrade of the low-level values in H
