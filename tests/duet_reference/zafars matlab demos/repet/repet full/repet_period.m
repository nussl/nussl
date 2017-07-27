%   Repeating period estimate from the beat spectrum for repet
%       p = repet_period(b);
%
%   Input(s):
%       b: beat spectrum [1 x #frames]
%
%   Output(s):
%       p: repeating period (in samples)
%
%   See also repet, repet_core, beat_spectrum, local_min_max

%   Author: Zafar RAFII (zafarrafii@u.northwestern.edu)
%   Last update: February 2011

function p = repet_period(b)

p = xcsm(b);                                                    % Cross-CSM of the beat spectrum
p = local_min_max(p(1:floor(end/2)),1,0);                       % Strict local minima & maxima
p = p{2};                                                       % Strict local maxima

p(1,:) = [];                                                    % First maxima is lag 0
p = sortrows(p,2);                                              % Sort the maxima
p = p(end,1)-1;                                                 % Period = lag index - 1
