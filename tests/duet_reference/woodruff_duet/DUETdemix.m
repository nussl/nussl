function [out,outSpec] = DUETdemix(sourceAlpha,sourceDelta,in,closeSource,fs,winLength,hopFactor,winType)

% [out,outSpec] = DUETdemix(sourceAlpha,sourceDelta,in,closeSource,fs,winLength,hopFactor,winType)
%
% This function implements the maximum likelihood demixing of the DUET
% algorithm, described in [Yilmaz,Rickard].  The function expects
% cross-channel characteristics to have previoulsy been determined (by
% 'DUETanalysis'), and distributes each time-frequency frame of a stereo
% mixture to the most likely source signal using the function
% 'likelihoodG'.
%
% Input Parameters
% ----------------------------------
% sourceAlpha(source_num) - the relative amplitude between channels for each source
%
% sourceDelta(source_num) - the relative delay between channels for each source
%
% in - two channel signal input
%
% closeSource - the distance in meters to the closest possible source (if
%               unknown, use a small value...say .1 meters or so)
%
% fs - the sampling frequency of the audio signals
%
% winLength - (must be a power of 2, try 1024 if not sure) the window 
%             length used in the STFT stage
%
% hopFactor - (should be an integer between 1 and 4) this controls the hop 
%             size (number of samples between STFT windows) as follows: 
%             hopSize = winLength/(2^hopFactor)
%
% winType - the type of amplitude window to use in FFT processing.  this is
%           expected to be a string containing a word that the Matlab 
%           'window' function will recognize.  type "help window" into the 
%           command line if unsure.  if this parameter is empty (i.e. 
%           winType = []), a hanning window is used.
%
% Output Parameters
% ----------------------------------
% out - the final time-domain source signal estimates
%
% outSpec(frame,bin,source_num) - the final source spectra estimates
%
% John Woodruff
% June 24, 2005


%calculate frequency bin spacing for 'likelihood2'
w0 = fs/winLength;

% calculate the STFT of the input signals
X = stft_(in,winLength,hopFactor,winType);

[numTimes,numFreqs] = size(X(:,:,1));
numSources = length(sourceAlpha);

% only use half of the symmetric spectrum for processing, and remove the DC
% component of the signal
numFreqs = numFreqs/2;
X = X(:,2:numFreqs+1,:);

% maximum possible distance (in meters) between mics
maxDistance = 340.29/fs;

% assume that sources can approach no closer than 'closeSource' meters from 
% mics.  so, maximum possible attenuation can be determined using distances 
% of 'closeSource' meters and 'closeSource'+'maxDistance' meters.  
maxAttenuation = (closeSource^2)/((closeSource+maxDistance)^2);
minAlpha = (maxAttenuation - 1/maxAttenuation)/2;


% find relative amplitude values for selected alpha values (convert
% symmetric attenuation back to amplitude attenuation)
sourceA = ((sourceAlpha*abs(minAlpha))+sqrt((sourceAlpha*abs(minAlpha)).^2+4))/2;

% find relative delay values for selected delta values
sourceD = sourceDelta/fs;


[outSpec,estimate] = likelihoodG(X,sourceA,sourceD,fs,winLength);

fileLength = length(in);

% inverse fft using the 'overlap and add' method
out = overlapAdd(outSpec,fileLength,winLength,hopFactor);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [outSpec,L] = likelihoodG(X,sourceA,sourceD,fs,winLength)

%
% [outSpec,L] = likelihoodG(X,sourceA,sourceD,fs,winLength)
%
% Input parameters
% --------------------
% X(t,f,x) - the complex valued time-frequency frames for x signals
%
% sourceA - the relative amplitude values associated with each source s
%
% sourceD - the relative phase delay associated with each source s
%
% fs - the sampling rate
%
% winLength - the length of the window used in the stft/istft
%
% Output values
% -----------------------
% L(t,f,s) - the likelihood that frame (t,f) belongs to source s
%
% masks - the time-frequency masked created for each source s
%
% outSpec - the symmetric frequency spectrum of the output signals
%
% author - John Woodruff, May 17, 2005.



% standard deviation for the likelihood function
sigma = 1;

% constant used in the likelihood function
constant = 1/(2*pi*(sigma^2));

% get the number of times and frequencies
[numTimes,numFreqs] = size(X(:,:,1));

% get the number of sources
numSources = length(sourceA);

% initialize matrix L(t,f,s) to all 0s
L = zeros(numTimes,numFreqs,numSources);

% initialize masking matrices to 0 for each source desired
masks = zeros(numTimes,numFreqs,numSources);

% create a matrix of frequency values
w0 = (fs/(numFreqs*2))*2*pi;
freqs = w0*[1:numFreqs];
freqMatrix = repmat(freqs,[numTimes 1]);

% calculate the likelihood that time-frequency frame (t,f) belongs to
% source s using the bivariate gaussian distribution
for s=1:numSources
    % multiply X1 by relative amplitude and phase delay
    relX1 = (sourceA(s)*exp(-i*sourceD(s)*freqMatrix)).*X(:,:,1);
    complexDiff = abs(relX1-X(:,:,2));
    L(:,:,s) = exp(-constant*complexDiff/(1+(sourceA(s)^2)));
end

% set the time-frequency frame of the mask associated with the source that
% has the maximum likelihood value to sAdd
for t=1:numTimes
    for f=1:numFreqs
        [maxVal,source] = max(L(t,f,:));
        if maxVal > 0
            masks(t,f,source) = 1;
        end
    end
end


DC = zeros(numTimes,1);

for h=1:numSources
    % create the masked version of X (with added DC) for use in ifft
    halfMaskedX(:,:,h) = masks(:,:,h).*X(:,:,1); 
    %+ (1/sourceA(h))*exp(i*sourceD(h)*freqMatrix).*X(:,:,2));
    maskedX(:,:,h) = cat(2,DC,halfMaskedX(:,:,h));
    negMaskedX(:,:,h) = fliplr(maskedX(:,:,h));
    outSpec(:,:,h) = cat(2,maskedX(:,:,h),negMaskedX(:,2:(winLength/2),h));
end


