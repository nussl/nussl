function SNR = DuetRefinement(sources, angle)

% create and separate the stere mixture using refined spatial cues
% 
% Input:
% sources - a matrix of audio files (must be the same length and be 
%           arranged so that [fileLength,numSources] = size(sources))
%
% angle - the angle in degree between two adjacent sources
% Output:
%     - SNR: source to noise Ratio
% 
% Author: Jinyu Han
% Date: Dec 22, 2008

% create the mixture
fs = 44100;
numOfSource = 3;
angleR = pi*angle/180;
coordinate = [1,0;cos(angleR),sin(angleR);cos(2*angleR),sin(2*angleR)];
[mix, alphaTrue, deltaTrue] = makeMix(sources, coordinate, fs);

% get the spectrum
winLength = 2048;
hopFactor = 2;
winType = 'hann';

[fullSpec, srcSpec] = stft(sources,winLength,hopFactor,winType);
% calculate the STFT of the input signals
[fullSpec, mixSpec] = stft(mix,winLength,hopFactor,winType);
[numTimes,numFreqs] = size(mixSpec(:,:,1));
numSignals = numOfSource;

% DUET analysis to get spacial cues
closeSource = .1;
HNRthresh = 0.95;
timeStep = (winLength/(2^hopFactor))/fs;
[histo,srcAlpha,srcDelta] = ...
    DUETanalysis(mixSpec,winLength,fs,numOfSource,closeSource,0);

% DUET demix
[initSig,initSpec] = ...
    DUETdemix(srcAlpha,srcDelta,mix,closeSource,fs,winLength,hopFactor,winType);

% Pitch Tracking
[srcF, srcMedF] = FFestimate(sources,srcSpec,fs,timeStep,HNRthresh);
[F, medF] = FFestimate(initSig,initSpec,fs,timeStep,HNRthresh);
f0 = fs/winLength;

changes = 1;
cnt = 1;
% iterate refine the alpha and delta
while changes > 0.1    
    harmMasks = buildMask(numTimes,numFreqs,numSignals,f0,F);
    alpha = srcAlpha;
    delta = srcDelta;

    for i = 1:numSignals
        nOverlapMask = ones(numTimes,numFreqs);
        for j =1:numSignals
            if j ~= i
                nOverlapMask = nOverlapMask.*(1-harmMasks(:,:,j));
            else
                if medF(i) ~= 0
                    nOverlapMask = nOverlapMask.*harmMasks(:,:,i);
                end
            end
        end
        nOverlapSpec(:,:,1) = mixSpec(:,:,1).*nOverlapMask;
        nOverlapSpec(:,:,2) = mixSpec(:,:,2).*nOverlapMask;
        
        [histo,srcAlpha(i),srcDelta(i)] = ...
            DUETanalysis(nOverlapSpec,winLength,fs,1,closeSource,0);
    end
    srcAlpha = sort(srcAlpha);
    srcDelta = sort(srcDelta);

    % DUET demix
    [refineSig,refineSpec] = ...
        DUETdemix(srcAlpha,srcDelta,mix,closeSource,fs,winLength,hopFactor,winType);

    changes = max(max(abs((srcAlpha - alpha)./(alpha+eps))),max(abs((srcDelta - delta)./(delta+eps))));
    if(srcAlpha == alpha) 
        if(srcDelta == delta)
            break;
        end
    end
    if (cnt == 5)
        break;
    end
    cnt = 1+cnt;
    [F, medF] = FFestimate(refineSig,refineSpec,fs,timeStep,HNRthresh);
end

% rearange the src vector to match the estimate
F0Diff = zeros(numOfSource);
for i=1:numOfSource
    F0Diff(:,i) = abs((srcMedF(i)-medF)/(srcMedF(i)+eps));
end
src = sources;
for i=1:numOfSource
    [x,xIndex] = min(F0Diff);
    [x,yIndex] = min(min(F0Diff));
    xIndex = xIndex(yIndex);
    sources(:,xIndex) = src(:,yIndex);
    F0Diff(:,yIndex) = inf;
    F0Diff(xIndex,:) = inf;
end

% normalize and save the estimation
for i=1:numOfSource
    initSig(:,i) = initSig(:,i)/max(abs(initSig(:,i)));
    refineSig(:,i) = refineSig(:,i)/max(abs(refineSig(:,i)));
    wavwrite(initSig(:,i), fs, strcat('init_',num2str(i)));
    wavwrite(refineSig(:,i),fs,strcat('final_',num2str(i)));
end

srcLen = length(sources);
estLen = length(initSig);
L = abs(srcLen-estLen);
zeroPad = zeros(L,numOfSource);
if(srcLen > estLen)
    initSig = cat(1,zeroPad,initSig);
    refineSig = cat(1,zeroPad,refineSig);
else
    sources = cat(1,zeroPad,sources);
end

[SNR(:,1), aveSNR(1)] =getSNR(sources,initSig);
[SNR(:,2), aveSNR(2)] =getSNR(sources,refineSig);

fprintf('initial estimation: %f\n',aveSNR(1));
fprintf('final estimation: %f\n',aveSNR(2));

function harmonicMasks = buildMask(numTimes,numFreqs,numSignals,f0,F)
% 
% construct the harmonic masks based on F0s
% author: Jinyu Han
% 

% initialize the masks to the size of the signals with all 0 values
harmonicMasks = zeros(numTimes,numFreqs,numSignals);
fs = 44100;
% for each output signal
for s=1:numSignals
    % first, find all of the interference peaks to include with signal 
    % s's sound event n
    for t=1:numTimes
        % get the fundamental estimate for this time frame
        Festimate = F(t,s);

        % if the fundamental frequency estimate is nonzero
        if Festimate > 60
            % find the highest harmonic possible given F and fs
            highHarm = floor((fs/2)/Festimate);
            % create array of harmonic frequencies (integer multiples
            % of F)
            possHarmonics = Festimate*[1:highHarm];
            harmWidth = Festimate*.005*[1:highHarm];

            % convert the frequencies to bin numbers
            possBins = round(possHarmonics/f0);
            keepBins = find(possBins + 2 <= numFreqs); 
            possBins = possBins(keepBins);
            keepBins = find(possBins - 2 > 0);
            possBins = possBins(keepBins);

            numHarms(t,s) = length(possBins);

            % set the harmonic masks to 1 at the bins associated with
            % integer multiples of F
            harmonicMasks(t,possBins,s) = 1;
            harmonicMasks(t,possBins+1,s) = 1;
            harmonicMasks(t,possBins-1,s) = 1;
            harmonicMasks(t,possBins+2,s) = 1;
            harmonicMasks(t,possBins-2,s) = 1;
        end
    end
end

function [SNR, ave] = getSNR(cleanSignals, final)
% cal Signal to noise ratio
% Author: Jinyu Han
% 

num = size(cleanSignals,2);
for s=1:num
    [cor, lag] = max(abs(xcorr(cleanSignals(:,s),final(:,s)))); 
    lag = lag - length(cleanSignals);
    if lag > 0           
        SNR(s) =...
            10*log10(sum(cleanSignals(1:end-lag,s).^2)./sum((cleanSignals(1:end-lag,s)...
            -final(1+lag:end,s)).^2));
    else
        lag = abs(lag);
        SNR(s) = 10*log10(sum(cleanSignals(1+lag:end,s).^2)./...
            sum((cleanSignals(1+lag:end,s)-final(1:end-lag,s)).^2));
    end
end
ave = mean(SNR);