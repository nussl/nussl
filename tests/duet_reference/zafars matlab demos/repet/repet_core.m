%   REpeating Pattern Extraction Technique (REPET) (core algorithm version)
%       M = repet_core(V,p,t);
%
%   Input(s):
%       V: magnitude spectrogram [#bin x #frame x #channels]
%       p: repeating period
%       t: tolerance factor
%
%   Output(s):
%       M: binary time-frequency mask [#bin x #frame x #channels]
%
%   See also repet, beat_spectrum, repet_period

%   Author: Zafar RAFII (zafarrafii@u.northwestern.edu)
%   Last update: February 2011

function M = repet_core(V,p,t)

[n,m,l] = size(V);
r = ceil(m/p);                                                  % Number of repeating segments

M = zeros(n,m,l);
for k = 1:l                                                     % Loop on the channels   
    V0 = V(:,:,k);                                              % Magnitude spectrogram of channel k
    V1 = [V0,nan(n,r*p-m)];                                     % Nan-padding to have an integer number of segments
    V1 = reshape(V1,[n*p,r]);                                   % Reshape such that the columns are the segments
    V1 = [median(V1(1:n*(m-(r-1)*p),1:r),2); ...                % Median of the parts repeating for all the r segments (including the last one)
        median(V1(n*(m-(r-1)*p)+1:n*p,1:r-1),2)];               % Median of the parts repeating only for the first r-1 segments (empty if m = r*p)
    V1 = reshape(repmat(V1,[1,r]),[n,r*p]);                     % Duplicate repeating segment model and reshape back to have [n x r*p]
    V1 = V1(:,1:m);                                             % Truncate to the original number of frames to have [n x m]
    
    Mk = zeros(n,m);
    Mk(V0-2*V1<=t) = 1;                                         % Binary time-frequency mask (0 = non-repeating & 1 = repeating)
%     M(abs(log(V0./V1))<=1) = 1;                                 % Binary time-frequency mask (0 = non-repeating & 1 = repeating)
    M(:,:,k) = Mk;
end
