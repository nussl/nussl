%   Local peak detection (vector or matrix) (modified version)
%   (For the min between distance, check peaks in ascending order)
%       [P,I,J] = local_peaks2(X,win,thresh,opt);
%
%   Input(s):
%       X: vector or matrix of real values
%       dist: minimal distance between two peaks (default: 1)
%       thresh: threshold above which peaks are considered (default: -Inf)
%       opt: option to display the peaks (default: 0, no display)
%
%   Output(s):
%       P: values of the local peaks
%       I: indices of the local peaks if vector, row indices if matrix
%       J: [] if vector, column indices if matrix
%
%   See also local_peaks, local_extrema, local_min_max, local_min, local_max

%   Author: Zafar RAFII (zafarrafii@u.northwestern.edu)
%   Last update: April 2011

function [P,I,J] = local_peaks2(X,dist,thresh,disp)

if nargin<4, disp = 0; end
if nargin<3, thresh = -Inf; end
if nargin<2, dist = 0; end

[P,I,J] = local_peaks(X,3,thresh,0);                            % All the local peaks and their corresponding indices
M = sortrows([P,I,J],-1);                                        % Sort the local peaks in descending order
l = numel(P);                                                   % Number of local peaks
Y = nan(size(X));

if isvector(X)
    
    Y(I) = P;                                                   % X with only the local peaks
    Y = padarray(Y(:),dist,NaN);                                % Padding at the boundaries
    P = [];
    I = [];
    J = [];
    for k = 1:l                                                 % Loop on the local peaks in descending order
        i = M(k,2);                                             % Index of the local peak in X
        Yk = Y((1:2*dist+1)+i-1);                               % Window centered around the local maxima k
        [val,ind] = max(Yk);
        if ind == dist+1                                        % If the center is the max of the window, keep the peak
            P = cat(1,P,val);
            I = cat(1,I,i);
        else                                                    % Else, discard it
            Y(i+dist) = NaN;
        end
    end
    
    M = sortrows([P,I],2);
    P = M(:,1);
    I = M(:,2);
    
    if disp ~= 0
        figure,
        plot(X);
        hold on
        plot(I,P,'ro')
        legend('input function','local peaks')
    end
    
else
    
    Y(I,J) = P;                                                 % X with only the local peaks
    Y = padarray(Y,dist,NaN);                                   % Padding at the boundaries
    P = [];
    I = [];
    J = [];
    for k = 1:l                                                 % Loop on the local peaks in ascending order
        i = M(k,2);                                             % Row index of the local peak in X
        j = M(k,3);                                             % Column index of the local peak in X
        Yk = Y((1:2*dist+1)+i-1,(1:2*dist+1)+j-1);              % Window centered around the local maxima k
        [val,ind] = max(Yk);
        if ind == dist+1                                        % If the center is the max of the window, keep the peak
            P = cat(1,P,val);
            I = cat(1,I,i);
            J = cat(1,J,j);
        else                                                    % Else, discard it
            Y(i+dist,j+dist) = NaN;
        end
    end
    
    M = sortrows([P,I,J],[2,3]);
    P = M(:,1);
    I = M(:,2);
    J = M(:,3);
    
    if disp ~= 0
        figure,
        mesh(X);
        figure,
        X = zeros(n,m);
        for i = 1:numel(P,1)
            X(I,J) = P;
        end
        imagesc(X)
    end
end
