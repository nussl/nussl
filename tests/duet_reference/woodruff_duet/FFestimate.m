function [F, medF] = FFestimate(timeSig,sigSpec,fs,timeStep,HNRthresh)

% [F,medF] = FFestimate(timeSig, sigSpec, fs, timeStep, HNRthresh)
% 
% timeSig - time domain signal
% 
% sigSpec - spectrogram
% 
% fs - frequency
% 
% timeStep - timeStep = (winLength/(2^hopFactor))/fs;
% set the time step of the praat_pd function to equal the time step of the
% original STFTs
% 
% HNRthresh
% 
% A fundamental frequency estimation based on Praat_pd estimates.
% by Jinyu Han, Jan, 2009

numTimes = size(sigSpec,1);
numSignals = size(timeSig,2);

% use Bartsch's praat_pd to estimate fundamental frequency,
% harmonics-to-noise ratio and amplitude envelope for each signal
for s=1:numSignals
    [F(:,s),HNR(:,s),time,amp(:,s)] = praat_pd(timeSig(:,s),fs,0,timeStep);
end

% make sure length of praat_pd estimates is equal to number of time frames
% in partial signal estimates
numFrames = length(F);
if numTimes > numFrames
    F(numFrames+1:numTimes,:) = repmat(F(numFrames,:),[numTimes-numFrames 1]);
    HNR(numFrames+1:numTimes,:) = repmat(HNR(numFrames,:),[numTimes-numFrames 1]);
    amp(numFrames+1:numTimes,:) = repmat(amp(numFrames,:),[numTimes-numFrames 1]);
elseif numTimes < numFrames
    F = F(1:numTimes,:);
    HNR = HNR(1:numTimes,:);
    amp = amp(1:numTimes,:);
end

skip = 10;

for s=1:numSignals
    % change F estimates to 0 when HNR is too low
    F(:,s) = F(:,s).*(sign(HNR(:,s)-HNRthresh)~=-1);
%     F(:,s) = F(:,s).*mod(mod(sign(HNR(:,s) - HNRthresh),3),2);

    % change F estimates to 0 when amp is too low
    lowAmp = max(amp(:,s))*.05;
    F(:,s) = F(:,s).*(sign(amp(:,s)-lowAmp)~=-1);

    % find the change in frequency between frames
    changeF(:,s) = diff(F(:,s));
    for t=2:numTimes
        if abs(changeF(t-1,s)) > F(t-1,s)*.1
            minDiff = min(abs(F(t-1,s) - F(t:min(numTimes,t+skip),s)));
            if minDiff <= F(t-1,s)*.1
                F(t,s) = F(t-1,s);
                changeF(t,s) = F(t+1,s)-F(t,s);
            end
        end
    end

    for t=1:numTimes
        % set all 0 values to the most correlated, 'reliable' neighbor
        if F(t,s)==0
            % find prior nonzero estimates
            earlyFrames = find(F(max(1,t-5):t,s));
            % find later nonzero estimates
            lateFrames = find(F(t:min(t+5,numTimes),s)) + (t-1);

            if isempty(earlyFrames)
                earlyCorr = -inf;
            else
                % get the closest prior frame to frame t
                earlyFrame = earlyFrames(length(earlyFrames));
                % calc. the correlation between frame t and 'earlyFrame'
                earlyCorr = xcorr(abs(sigSpec(earlyFrame,:,s)),abs(sigSpec(t,:,s)),0);
            end

            if isempty(lateFrames)
                lateCorr = -inf;
            else
                % get the closest later frame to frame t
                lateFrame = lateFrames(1);
                % calc. the correlation between frame t and 'lateFrame'
                lateCorr = xcorr(abs(sigSpec(lateFrame,:,s)),abs(sigSpec(t,:,s)),0);
            end

            % if either correlation was calculated
            if (earlyCorr > -inf) && (lateCorr > -inf)
                % set the fundamental frequency of frame t to the most
                % correlated nonzero neighboring frame
                if earlyCorr > lateCorr
                    F(t,s) = F(earlyFrame,s);
                else
                    F(t,s) = F(lateFrame,s);
                end 
            end
        end
    end
    
    % find the change in frequency between frames
    changeF(:,s) = diff(F(:,s));

    % if the frequency change between frames is too large, check to see if
    % any of the next three frames return to the frequency at t-1.  if so,
    % change the frequency at t to match t-1.  if not, do not adjust
    % frequency estimate
    timeWindowLength = timeStep;
    skip = ceil(.06/timeWindowLength);
    
    for t=2:numTimes
        if abs(changeF(t-1,s)) > F(t-1,s)*.1
            minDiff = min(abs(F(t-1,s) - F(t:min(numTimes,t+skip),s)));
            if minDiff <= F(t-1,s)*.1
                F(t,s) = F(t-1,s);
                changeF(t,s) = F(t+1,s)-F(t,s);
            end
        end
    end
end

for k=1:numSignals
    if(var(F(F(:,k)>0,k)) > 10)
        for p=1:numSignals
            if(var(F(F(:,p)>0,p))<10)
                index = mod(F(:,k)./mean(F(F(:,p)>0,p)),1)<0.01 |...
                    (1-mod(F(:,k)./mean(F(F(:,p)>0,p)),1))<0.01;
                F(index,k) = 0;
            end
        end
    end
end

% calculate the average value for the pitch
for k=1:numSignals
    medF(k) = sum(F(:,k))/sum(F(:,k)>0);
    if(sum(F(:,k)>0) == 0)
        medF(k) = 0;
    end
end

% if the frequency is far from the average value, check to see if it's
% close to any of the other source, if so, move it to the other source. If
% not, change it to zero.
% find((F(:,s) - medF(s)) > 0.15*medF(s));
% for t=1:numTimes
%     for s=1:numSignals
% %         if F(t,k)
%     end
% end