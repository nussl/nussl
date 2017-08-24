function [mix,sourceAlpha,sourceDelta,closeSource,truncSources] = makeMix(sources,coordinates,fs)

% [mix,closeSource,truncSources,sourceAlpha,sourceDelta] = makeMix(sources,coordinates,fs)
%
% This function is used to make a stereo mix which mimics an anechoic stereo
% recording using two, closely spaced omni-directional microphones.
% Cartesian coordinates are entered to specify source locations relative to
% microphone location.  Microphone spacing is determined based on the
% sampling rate used ('fs').  
%
% Input parameters
% --------------------
% sources - a matrix of audio files (must be the same length and be 
%           arranged so that [fileLength,numSources] = size(sources))
% size(sources) returns x-number of rows, y-number of columns
%
% coordinates - a matrix of coordinate pairs that represent the position on
%               the x-y coordinate plane, assuming that microphone 1 is 
%               located at (0,0). values are in meters. (e.g. for 3
%               instruments, this could be [-1, 0; 0, 0.5; 1, 0])
%
% fs - the sampling rate of the audio files
%
% Output values
% -----------------------
% mix - a stereo mix of all of the sources
%
% closeSource - the distance to the closest of the sources in meters
%
% truncSources - truncated versions of the source files to match the length
%               of the output vectors
%
% sourceAlpha - the relative amplitude values associated with each source s
%               (these values are the same as those plotted in the y-axis 
%               of the 'mixing parameters' plot in the demo)
%
% sourceDelta - the relative delay values associated with each source s
%               (these values are the same as those plotted in the x-axis 
%               of the 'mixing parameters' plot in the demo)
%
% author - John Woodruff, June 1, 2005.
% additional tweaks and corrections by Jinyu Han, Jan 2009


% the factor to umpsample by, therefore, sample delays with be .1, .2,
% .3..etc. samples long.  
sampQuantize = 10;

[fileLength,numSources] = size(sources);

% maximum delay is 1 sample...so maximum delay time is 1/fs
maxDelay = 1/fs;

% to calculate maximum distance, multiply max delay by the speed of sound
maxDistance = maxDelay*340.29;

% set the theoretical microphone locations
mic1 = [-maxDistance/2, 0];
mic2 = [maxDistance/2, 0];

% calculate the relative amplitude and delay times between the two mics for
% each source
for i=1:numSources
    % the coordinates of this source
    xyPair = coordinates(i,:);
    % distance to microphone 1 (in meters)
    dist1(i) = sum((xyPair - mic1).^2)^0.5;
    dist2(i) = sum((xyPair - mic2).^2)^0.5;
    % minimum distance to a microphone
    minDist(i) = min(dist1(i),dist2(i));
    % the difference between the two distances
    relDist(i) = dist2(i) - dist1(i);
    % calculate delay by dividing distance (in meters) by speed of sound
    relDelay(i) = (relDist(i)/340.29)/maxDelay;
    
%     The phase difference from the second microphone to the first
%     microphone
    sourceDelta(i) = round(relDelay(i)*sampQuantize)/sampQuantize;
    % quantize 'relDelay' into quarters of a sample
    sampDelay(i) = round(relDelay(i)*sampQuantize);
   
    % resample audio files to enable delay shifts of less than 1 sample
    interpSources(:,i) = interp(sources(:,i),sampQuantize);
end

% store the closest source distance to the microphones (in order to
% normalize the relative amplitude attenuation values
closeSource = min(minDist);

% maxAttenuation and maxAlpha are not used
% calculate the maximum attenuation possible given the closest source
% maxAttenuation = (closeSource^2)/((closeSource+maxDistance)^2);

% calculate the maximum alpha possible
% maxAlpha = (maxAttenuation - 1/maxAttenuation)/2;

% set the source amplitudes relative to the closest source
for i=1:numSources
%     relAmp(:,i) = [((closeSource^2)/(dist1(i)^2))^0.5, ((closeSource^2)/(dist2(i)^2))^0.5];
    relAmp(:,i) = [((closeSource^2)/(dist1(i)^2)), ((closeSource^2)/(dist2(i)^2))];
%     amplitute ratio between the second microphone and the first
%     microphone
    sourceAmp(i) = relAmp(1,i)/relAmp(2,i);
end

% find source alpha value and normalize by maxAlpha
sourceAlpha = sourceAmp - 1./sourceAmp;

% Not necessay
% sourceAlpha = sourceAlpha/abs(maxAlpha);


% sourceDelta = fliplr(sourceDelta) ;

% the maximum sample delay of the sources
maxSampDelay = max(abs(sampDelay));

% initilize arrays to store the mixture signals
tempMix = zeros((fileLength*sampQuantize)-(2*maxSampDelay),2);

for i=1:numSources
    % if the sample delay is greater than 0 (source is closer to mic 1 than
    % it is to mic 2)
    if sampDelay(i) > 0
        % set the starting and ending sample values of this source in each
        % mixture
        start1 = 1+maxSampDelay;
        stop1 = (fileLength*sampQuantize)-maxSampDelay;
        start2 = 1+maxSampDelay+sampDelay(i);
        stop2 = (fileLength*sampQuantize)-maxSampDelay+sampDelay(i);
    % the source is equidistant from the mics
    elseif sampDelay(i)==0
        % set the starting and ending sample values of this source in each
        % mixture
        start1 = floor(1+(maxSampDelay*3/2));
        stop1 = floor((fileLength*sampQuantize)-(maxSampDelay/2));
        start2 = start1;
        stop2 = stop1;
    % if sample delay is less than 0 (source is closer to mic 2 than mic 1)
    else
        % set the starting and ending sample values
        start1 = 1+maxSampDelay-sampDelay(i);
        stop1 = (fileLength*sampQuantize)-maxSampDelay-sampDelay(i);
        start2 = 1+maxSampDelay;
        stop2 =  (fileLength*sampQuantize)-maxSampDelay;
    end
    % create arrays that store the original sources, delayed and scaled as
    % they would be in mixture 1
    trunc(:,i) = interpSources(start1:stop1,i)*relAmp(1,i);
    % scale each source by the relative amplitude value and add the source
    % to the mixture
    tempMix(:,1) = tempMix(:,1) + interpSources(start1:stop1,i)*relAmp(1,i);
    tempMix(:,2) = tempMix(:,2) + interpSources(start2:stop2,i)*relAmp(2,i);
end

% resample the mixtures to get the original sampling rate
for i=1:2
    mix(:,i) = resample(tempMix(:,i),1,sampQuantize);
end

% resample the shifted and scaled source signals
for i=1:numSources
    truncSources(:,i) = resample(trunc(:,i),1,sampQuantize);
end

maxOut = max(max(max(mix)),abs(min(min(mix))));

mixLength = length(mix);

    