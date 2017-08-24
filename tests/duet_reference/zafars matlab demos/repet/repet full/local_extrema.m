%   Local extrema finder (for a vector)
%       E = local_extrema(x,strict,disp);
%
%   Input(s):
%       x: vector of (real) values
%       strict: option to only identify the strict local extrema (default: 0, all the local extrema)
%       disp: option to plot the extrema (default: 0, no plot)
%
%   Output(s):
%       E: matrix of the extrema indices (:,1) and corresponding values (:,2)
%
%   See also local_min_max, local_peaks

%   Author: Zafar RAFII (zafarrafii@u.northwestern.edu)
%   Last update: October 2010

function E = local_extrema(x,strict,disp)

if ~isvector(x)
    error('The input must be a vector.');
end

if nargin<3, disp = 0; end
if nargin<2, strict = 0; end

x = x(:);
n = length(x);
i = diff([x(2);x;x(n-1)]);                                      % Discrete derivatives to find sign changes corresponding to extrema
if strict == 0
    i = find(i(1:end-1).*i(2:end)<=0);                          % Indices of the changing signes (including zeros)
else
    i = find(i(1:end-1).*i(2:end)<0);                           % Indices of the strict changing signes (not including zeros)
end
E = [i,x(i)];                                                   % Extrema indices and corresponding values (including extremities)

if disp ~= 0
    figure, plot(x)
    hold on
    plot(E(:,1),E(:,2),'ro')
    legend('input function','local extrema')
end
