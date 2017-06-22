function x = tfsynthesis(timefreqmat,swin,timestep,numfreq)     %#ok<INUSD>

% time-frequency synthesis
% TIMEFREQMAT is the complex matrix time-freq representation
% SWIN is the synthesis window
% TIMESTEP is the # of samples between adjacent time windows.
% NUMFREQ is the # of frequency components per time point.
%
% X contains the reconstructed signal.

swin = swin(:);                                                 % make synthesis window go columnwise
winlen = length(swin);
[numfreq numtime] = size(timefreqmat);
ind = rem((1:winlen)-1,numfreq)+1;
x = zeros((numtime-1)*timestep+winlen,1);

for i=1:numtime                                                 % overlap, window, and add
    temp = numfreq*real(ifft(timefreqmat(:,i)));
    sind = ((i-1)*timestep);
    rind = (sind+1):(sind+winlen);
    x(rind) = x(rind)+temp(ind).*swin;
end
