function out = overlapAdd(X,fileLength,winLength,hopFactor)

% out = overlapAdd(X,fileLength,winLength,hopFactor)
%
% Input parameters
% --------------------
% X - Time-frequency domain representation of signals, X(t,f,s)
%
% fileLength - the length (in samples) of the desired output signals
%
% winLength - the length of the ifft window
%
% hopFactor - the hop size used is winLength/(2^hopFactor)
%
% Output values
% -----------------------
% out - the time-domain output signals
%
% author - John Woodruff, May 17, 2005.


hop = (winLength/(2^hopFactor));

% get the appropriate number of signals, sources, and time-frequency frames
[numTimes,numFreqs,numSources] = size(X);

out = zeros(fileLength,numSources);

for h=1:numSources
    for i=1:numTimes
        winLeft = 1 + ((i-1)*hop);
        winRight = winLeft + winLength - 1;
        outFrame = ifft(X(i,:,h),winLength,'symmetric');
        if winRight > fileLength, winRight = fileLength; end;
        out(winLeft:winRight,h) = out(winLeft:winRight,h) + outFrame(1:winRight-winLeft+1)';            
    end
end

out = out/(2^(hopFactor-1));