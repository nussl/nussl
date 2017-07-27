%   Cosine Similarity Measure
%       val = csm(X,Y);
%
%   Input(s):
%       X: original data matrix or vector
%       Y: data matrix or vector to compare with
%
%   Output(s):
%       value: CSM value [-1,1]
%              (-1: opposite, 0: orthogonal, 1: similar)
%
%   See also snr, sdr

%   Author: Zafar RAFII (zafarrafii@u.northwestern.edu)
%   Last update: October 2010

function val = csm(X,Y)

X = X(:);
Y = Y(:);
val = X'*Y/(norm(X,2)*norm(Y,2));
