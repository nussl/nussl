function [X, hX] = stft(in,winLength,hopFactor,winType,varargin)

% X = stft(in,winLength,hopFactor,fs)
%
% Input parameters
% --------------------
% in - two channel signal input
%
% winLength - the length of the ifft window
%
% hopFactor - hop = (winLength/(2^hopFactor))
% 
% winType - type of the window, if don't know use 'hann'
% 
% fs - frequency of the input, used to plot the spectrogram 
% 
% Output values
% -----------------------
% X(t,f,x) - the complex valued time-frequency frames for x signals
%
% hX - half of the symmetric spectrogram
% 
% author - John Woodruff, May 25, 2005.
% Addtional tweaks by Jinyu Han, Jan, 2009

% create window vector for use in specgram
win = window(winType,winLength);

% find the correct hop size
hop = (winLength/(2^hopFactor));

[fileLength,numSignals] = size(in);
numsteps = floor(fileLength/hop);

% for each channel, calculate the STFT using specgram.  transpose matrix to
% end up with X(time,frequency,channel).
for x=1:numSignals
    for i=1:numsteps
        winLeft = 1 + ((i-1)* hop);
        winRight = winLeft + winLength  - 1;
        if winRight < fileLength
            signal = in(winLeft:winRight,x);
            signal = signal.*win;
        else
            signal = in(winLeft:fileLength,x);
            signal = signal.*win(1:fileLength-winLeft+1);
        end
        X(i,:,x) = fft(signal,winLength);
    end
end
hX = X(:,2:winLength/2+1,:);

if(nargin==5)
    fs = varargin{1};
    minFreq = 50;
    maxFreq = 5000;
    binFreqs = [1:winLength/2]*(fs/winLength);
    useBins = find(binFreqs < maxFreq & binFreqs > minFreq);
    for i=1:numSignals
        figure;
        surf(abs(hX(:,useBins,i)));
        xticks = get(gca,'xtick');
        xtickLabels = binFreqs(xticks+1);
        set(gca,'xticklabel',xtickLabels);
    end
end