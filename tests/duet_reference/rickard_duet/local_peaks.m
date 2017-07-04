%   Local peak detection (vector or matrix)
%       [P,I,J] = local_peaks(X,win,thresh,opt);
%
%   Input(s):
%       X: vector or matrix of real values
%       win: sliding window size (default: 3 if vector, [3x3] if matrix)
%            (even lengths will be rounded to the nearest odd integer toward minus infinity)
%       thresh: threshold above which peaks are considered (default: -Inf, all the peaks are considered)
%       disp: option to display the peaks (default: 0, no display)
%
%   Output(s):
%       P: values of the local peaks
%       I: indices of the local peaks if vector, row indices if matrix
%       J: [] if vector, column indices if matrix
%
%   See also local_extrema, local_min_max, local_min, local_max

%   Author: Zafar RAFII (zafarrafii@u.northwestern.edu)
%   Last update: April 2011

function [P,I,J] = local_peaks(X,win,thresh,disp)

if nargin<4, disp = 0; end
if nargin<3, thresh = -Inf; end
if nargin<2
    if isvector(X);
        win = 3;
    else
        win = [3,3];
    end
end

if isvector(X)
    
    n = length(X);
    if win <= 0
        error('The window length is too short.')
    elseif win > n
        error('The window length is too large.')
    end
    
    win = ceil(win/2)*2-1;                                      % Round the window size to the nearest odd integer toward minus infinity
    v = floor(win/2);
    
    X = padarray(X(:),v,NaN);                                   % Padding X with NaN values at the boundaries for the sliding window
    P = [];
    I = [];
    J = [];
    for i = 1:n
        Xi = X((1:win)+i-1);                                    % Sliding window of size win
        [val,ind] = max(Xi);
        if ind == v+1 && val >= thresh                          % If maximum local & above threshold, it is a peak
            P = cat(1,P,val);
            I = cat(1,I,i);
        end
    end
    
    if disp ~= 0
        figure,
        plot(X((1:n)+v));                                       % Un-padded version of X
        hold on
        plot(I,P,'ro')
        legend('input function','local peaks')
    end
    
else
    
    [n,m] = size(X);
    if any(win <= 0)                                            % Error, if any of the window dimensions is smaller than 0
        error('The window size is too small.')
    elseif any(win > [n,m])                                     % Error, if any of the window dimensions is bigger than the size of the input
        error('The window size is too big.')
    end
    
    win = ceil(win/2)*2-1;
    win1 = win(1);
    win2 = win(2);
    v = floor(win/2);
    v1 = v(1);
    v2 = v(2);
    
    X = padarray(X,v,NaN);
    P = [];
    I = [];
    J = [];
    for i = 1:n
        for j = 1:m
            Xij = X((1:win1)+i-1,(1:win2)+j-1);                 % Sliding window of size win
            [val, idx] = max(Xij(:));
            [row, col] = ind2sub(size(Xij),idx);

            if all([row,col] == [v1+1,v2+1]) && val >= thresh   % If maximum local & above threshold, it is a peak
                P = cat(1,P,val);
                I = cat(1,I,i);
                J = cat(1,J,j);
            end
        end
    end
    
    if disp ~= 0
        figure,
        mesh(X((1:n)+v1,(1:m)+v2));                             % Un-padded version of X
        figure,
        X = zeros(n,m);
        for i = 1:numel(P,1)
            X(I,J) = P;
        end
        imagesc(X)
    end
    
end
