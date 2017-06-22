%   Cross-Cosine Similarity Measure
%       c = xcsm(varargin);
%
%   Input(s):
%       x: vector 1, (if x only, auto-csm of x)
%       y: vector 2, (if y included, cross-csm between x & y)
%
%   Output(s):
%       c: cross-csm vector
%
%   See also xcorr, csm, repet_period

%   Author: Zafar RAFII (zafarrafii@u.northwestern.edu)
%   Last update: February 2011

function c = xcsm(varargin)

k = numel(varargin);
switch k
    case 1                                                      % Auto-CSM
        
        x = varargin{1};
        n = length(x);
        c = zeros(n,1);
        for i = 0:n-1
            c(i+1) = csm(x(1:n-i),x(i+1:n));
        end
        
    case 2                                                      % Cross-CSM
        
        x = varargin{1};
        n = length(x);
        x = x(:);
        y = varargin{2};
        m = length(y);
        y = y(:);
        
        if n > m                                                % if x & y have different lengths, the shortest one is zero-padded
            y = [y;zeros(n-m,1)];
        elseif n < m
            x = [x;zeros(m-n,1)];
            n = m;
        end
        
        c = zeros(2*n-1,1);
        for i = 0:n-1
            c(i+1) = csm(x(1:n-i),y(i+1:n));
        end
        for i = 1:n-1
            c(n+i) = csm(x(i+1:n),y(1:n-i));
        end
        
end
