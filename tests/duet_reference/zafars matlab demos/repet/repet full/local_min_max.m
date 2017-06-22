%   Local extrema (maxima & minima) finder (for a vector)
%       E = local_min_max(x,strict,disp);
%
%   Input(s):
%       x: vector of (real) values
%       strict: option to only identify the strict local extrema (default: 0, all the local extrema)
%       disp: option to plot the extrema (default: 0, no plot)
%
%   Output(s):
%       E: cell of the matrices of the extrema indices (:,1) and corresponding values (:,2)
%          E{1}: local minima matrix
%          E{2}: local maxima matrix
%
%   See also local_extrema, local_peaks

%   Author: Zafar RAFII (zafarrafii@u.northwestern.edu)
%   Last update: October 2010

function E = local_min_max(x,strict,disp)

if ~isvector(x)
    error('The input must be a vector.');
end

if nargin<3, disp = 0; end
if nargin<2, strict = 0; end

E = local_extrema(x,strict,0);                                  % Local extrema (minima & maxima)
e = E(:,2);                                                     % Extrema values
n = [[e(1);e(1:end-1)],[e(2:end);e(end)]];                      % Extrema neighbors (left & right)

m = min(n,[],2);
j = e<=m;
M1 = E(j,:);                                                    % Minima indices and corresponding values
m = max(n,[],2);
j = e>=m;
M2 = E(j,:);                                                    % Maxima indices and corresponding values

E = {M1,M2};

if disp == 1
    figure, plot(x)
    hold on
    plot(M1(:,1),M1(:,2),'gx')
    plot(M2(:,1),M2(:,2),'ro')
    legend('input function','local minima','local maxima')
end
