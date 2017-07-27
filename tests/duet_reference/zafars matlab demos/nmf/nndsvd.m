% Non-Negative Double Singular Value Decomposition (NNDSVD)
%   [W,H,S] = nndsvd(V,r,option);
%
% Input:
%   V: matrix to analyze V ~= W*S*H  [n x m]
%   r: number of components (default: r=20 components)
%   option: =1, upgrade of the low-level components (default: =0, no upgrade)
%
% Output:
%   W: matrix of r non-negative "output" basis vectors for V [n x r] = initial basis matrix for NMF
%   H: matrix of r non-negative "input" basis vectors for V [r x m] = initial weight matrix for NMF
%   S: diagonal matrix of the singular values [r x r] = initial core matrix (if 3 output arguments)

function [W,H,S] = nndsvd(V,r,option)

if nargin<3, option=0; end
if nargin<2, r=20; end

% Truncated Singular Value Decomposition from V for the r best components:

[n,m] = size(V);
v = sum(V(:));
[U,S,T] = svds(V,r);                                            %Partial SVD
T = T';

% NNDSVD algorithm:

W = zeros(n,r);
H = zeros(r,m);

W(:,1) = abs(U(:,1));                                           %1st component of W
H(1,:) = abs(T(1,:));                                           %1st component of H

for i=2:r
    [up,un] = posneg(U(:,i));                                   %For each component, U is broken up into its positive & negative matrices
    n_up = norm(up,2);                                          %Norm 2 of up
    n_un = norm(un,2);                                          %Norm 2 of un
    
    [tp,tn] = posneg(T(i,:));                                   %For each component, T is broken up into its positive & negative matrices
    n_tp = norm(tp,2);                                          %Norm 2 of tp
    n_tn = norm(tn,2);                                          %Norm 2 of tn
    
    n_p = n_up*n_tp;                                            %Positive overall norm 2 for the ith component
    n_n = n_un*n_tn;                                            %Negative overall norm 2 for the ith component
    
    if (n_p>n_n)
        W(:,i) = sqrt(n_p)*up/n_up;
        H(i,:) = sqrt(n_p)*tp/n_tp;
    else
        W(:,i) = sqrt(n_n)*un/n_un;
        H(i,:) = sqrt(n_n)*tn/n_tn;
    end
end

% Filtered components:

if option==1
    o = ones(1,r);
    v = sum(V,2)*sum(W(:))/s;                                   %Normalized filter to increase low values of W (sum(W(:))=r)
    W = W+1e-3*v*o;                                             %Increase of the low level values
    v = sum(V,1)*sum(H(:))/s;                                   %Normalized filter to increase low values of H
    H = H+1e-3*o'*v;                                            %Increase of the low level values
end

WSH = W*S*H;
S = S*v/sum(WSH(:));                                            %Increase of the overall energy
if nargout==2                                                   %If 2 output arguments: W & H
    S = sqrt(S);
    W = W*S;                                                    %Transfer half of the energy from S to W
    H = S*H;                                                    %Transfer half of the energy from S to H
end
