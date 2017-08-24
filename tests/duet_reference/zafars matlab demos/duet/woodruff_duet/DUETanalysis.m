function [H,sourceAlpha,sourceDelta] = DUETanalysis(mixSpec,winLength,...
    fs,numPeaks,closeSource,print)

% [H,sourceAlpha,sourceDelta] = DUETanalysis(in,winLength,fs,numPeaks,closeSource,print)
%
% This function analyzes an anechoic, stereo mixture to deterine the
% cross-channel characteristics of each source using the DUET algorithm
% described in [Yilmaz,Rickard].  
% 
% The function expects a mixture spectra ('X'), the number of sources to 
% find ('numPeaks') and an estimate of the closest possible source to the 
% microphones ('closeSource').  
% 
% It then creates a two-dimensional histogram of cross-channel amplitude scaling
% and time-shift differences ('H') and searches this histogram for peaks,
% which are returned in the arrays 'sourceAlpha' and 'sourceDelta'.  

% the Yilmaz/Rickard implementation uses a k-means clustering algorithm to
% determine peaks.  This method simply searches for the largest peaks in
% the smoothed historgram.  Also note that the subfunction, 'make2dHist' 
% was provided by Scott Rickard and adapted for use by John Woodruff.
%
% Input parameters
% --------------------
% mixSpec - mixture spectra (half of the symmetric spectrum)
%
% winLength - (must be a power of 2, try 1024 if not sure) the window 
%             length used in the STFT stage
%
% hopFactor - (should be an integer between 1 and 4) this controls the hop 
%             size (number of samples between STFT windows) as follows: 
%             hopSize = winLength/(2^hopFactor)
%
% fs - the sampling frequency of the audio signals
%
% numPeaks - number of peaks to find in the histogram (in most cases, this
%            should equal the number of sources in the mix)
%
% closeSource - the distance in meters to the closest possible source (if
%               unknown, use a small value...say .1 meters or so)
%
% print - 0 or 1 to turn printing off or on
%
% Output values
% -----------------------
%
% H(a,d) - weighted and smoothed histogram of the number of time-frequency
%          frames with mixing parameters (alpha,delta)
%
% sourceAlpha - the log-relative amplitude values associated with each source s
%               (these values should match peaks on the histogram H)
%
% sourceDelta - the relative phase delay associated with each source s
%               (these values should match peaks on the histogram H)
%
% author - John Woodruff, May 5, 2005.
% additional tweaks by Jinyu Han, Jan, 2009

X = mixSpec;
[numTimes,numFreqs] = size(mixSpec(:,:,1));

% calculate the relative transform of X1/X2 (use 'eps' to add smallest
% possible float t(o both, prevents dividing by 0)
R = (X(:,:,2)+eps)./(X(:,:,1)+eps);

% calculate the amplitude for each frame in R.  'amp' is a matrix of
% relative amplitudes between input channel 1 and 2 for each time-frequency
% frame
amp = abs(R);

% calculate the phase delay for each frame in R.  'angle' is a matrix of
% relative phase delays between input channel 1 and 2 for each
% time-frequency frame
phase = angle(R);

% calculate the frequency bin spacing
w0 = fs/winLength;

% for each time-frequency frame, divide the phase delay by the frequency of
% that frame.
freqs = w0*2*pi*[1:numFreqs];
freqMatrix = repmat(freqs,[numTimes 1]);
delta = (-1./freqMatrix).*phase;

% create symmetric attenuation
alpha = amp - 1./amp;

% maximum usable distance the distance travelled during 1 sample, so this
% is set to the speed of sound divided by sampling rate
maxDistance = 340.29/fs;

% assume that sources can approach no closer than the 'closeSource' distance
% from mics.  so, maximum possible attenuation can be determined using 
% distances of closeSource and closeSource+maxDistance cm.  
maxAttenuation = (closeSource^2)/((closeSource+maxDistance)^2);
minAlpha = (maxAttenuation - 1/maxAttenuation)/2;

% normalize alpha to have major peaks between -1 and 1
% alpha = alpha/abs(minAlpha);

% normalize delta to have major peaks between -1 and 1 samples
delta = delta*fs;

% call the 'make2dHist' function to create a 2-dimensional weighted
% histogram (smoothed and unsmoothed) of alpha and delta estimates. 
histRes = 500;
% histRes = 230;
[H,h,alphaV,deltaV,tfweight] = make2dHist(X,alpha,delta,histRes,histRes);

aBin = [1:histRes];
dBin = [1:histRes];

if print~=0
    % draw the histograms as a 3-d surfaces
    % unsmoothed
%     figure(1);
%     surf(dBin,aBin,h);
%     xlabel('delay');
%     ylabel('amplitude');
%     set(gca,'xtick',cat(2,1,round([0.16 0.33 0.5 0.66 0.83 1]*histRes)),'xticklabel',[-1.5 -1 -.5 0 .5 1 1.5],'ytick',cat(2,1,round([0.16 0.33 0.5 0.66 0.83 1]*histRes)),'yticklabel',[-1.5 -1 -.5 0 .5 1 1.5]);

    % smoothed
    figure;
%     surf(dBin,aBin,H);
    mesh(H);
    xlabel('Phase Difference');
    ylabel('Amplitude Ratio');
    set(gca,'xtick',cat(2,1,round([0.16 0.33 0.5 0.66 0.83 1]*histRes)),...
        'xticklabel',[-1.5 -1 -.5 0 .5 1 1.5],...
        'ytick',cat(2,1,round([0.16 0.33 0.5 0.66 0.83 1]*histRes)),...
        'yticklabel',[-1.5 -1 -.5 0 .5 1 1.5]);
end

coords = [deltaV(:),alphaV(:)];

% find standard deviation of matrix H
sdH = mean(var(H)).^0.5;

% create a new matrix setting all values below 'sdH' to 0.
% tH = (H > sdH).*H;
tH = (H > sdH).*H;

% use the 'imregionalmax' function to find all local maxima of the
% thresholded matrix, tH
maxTH = imregionalmax(tH);
dist = 6;
peaks = findPeaks(tH,maxTH,aBin,dBin,dist);

% find the highest peaks for the number of desired sources
for i=1:size(peaks,2)
    maxes(i) = H(peaks(1,i),peaks(2,i));
end

% numPeaks = size(peaks,2);

for i=1:numPeaks
    [peakValue,index] = max(maxes);
%     peak(1,i) = peaks(1,index);
%     peak(2,i) = peaks(2,index);
    sourceAlpha(i) = aBin(peaks(1,index));
    sourceDelta(i) = dBin(peaks(2,index));
    maxes = mod(maxes,peakValue-1);
end

sourceAlpha = (sourceAlpha*(3/histRes))-1.5;
sourceDelta = (sourceDelta*(3/histRes))-1.5;

sourceAlpha = sort(sourceAlpha);
sourceDelta = sort(sourceDelta);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function  [H,h,alphaV,deltaV,tfweight] = make2dHist(X,alpha,delta,alphaRes,deltaRes)

% [H,h,alphaV,deltaV,tfweight] = make2dHist(X,alpha,delta,alphaRes,deltaRes)
%
% Input parameters
% --------------------
% X(t,f,x) - the complex valued time-frequency frames for x signals
%
% alpha(t,f) - the relative amplitude difference between signal x1 and x2
%
% delta(t,f) - the relative phase delay between signal x1 and signal x2
%
% alphaRes - the number of bins for alpha in the histograms
%
% deltaRes - the number of bins for delta in the histograms
%
% Output values
% -----------------------
% H - weighted and smoothed histogram of the number of time-frequency
%     frames with mixing parameters (alpha,delta)
%
% h - weighted histogram, unsmoothed version of H
%
% author - Scott Rickard (adapted by John Woodruff)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calculate weighted histogram in alpha-delta space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%powers used to weight histogram and calculate histogram limits
tfweight = (abs(X(:,:,1)).*abs(X(:,:,2))).^0.5;

%set boundaries of histogram in alpha and delta direction to include only
%those values which contain the specified percentage of the tfweight
maxalpha = 1.5;
maxdelta = 1.5;
amask=(abs(alpha)<maxalpha)&(abs(delta)<maxdelta);
alphaV = alpha(amask);
deltaV = delta(amask);
tfweight = tfweight(amask);
%determine histogram indices
alphaI = round(1+(alphaRes-1)*(alphaV+maxalpha)/(2*maxalpha));
deltaI = round(1+(deltaRes-1)*(deltaV+maxdelta)/(2*maxdelta));

% create 2d weighted histogram
h = full(sparse(alphaI,deltaI,tfweight,alphaRes,deltaRes)); 

% smooth the histogram
H = twoDsmooth(h,round(.03*alphaRes));

function smat = twoDsmooth(mat,ker)
%TWO2SMOOTH - Smooth 2D matrix.
% S. Rickard and C. Fearon 
% Copyright 2005 by University College Dublin. 
%  
% smat = twoDsmooth(mat,ker)
%
% MAT is the 2D matrix to be smoothed.
% KER is either
%  (1) a scalar, in which case a ker-by-ker matrix of
%      1/ker^2 is used as the matrix averaging kernel
%  (2) a matrix which is used as the averaging kernel.
% 
% If either kernel matrix dimension is even, a zero
% vector is appended to either side of the even dimension
% and the pairwise average values are used along that dim.
% Thus, the kernel used always has both dimensions odd.
%
% The input matrix is extended with copies of the outside 
% border rows and columns so that the kernel matrix can be 
% centered on each value of the original input matrix.
%
% SMAT is the smoothed matrix (same size as mat).

if prod(size(ker))==1, % if ker is a scalar
    kmat = ones(ker,ker)/ker^2;
else 
    kmat = ker; 
end

% make kmat have odd dimensions
[kr kc] = size(kmat);
if rem(kr,2) == 0,
    kmat = conv2(kmat,ones(2,1))/2;
    kr = kr + 1;
end
if rem(kc,2) == 0,
    kmat = conv2(kmat,ones(1,2))/2;
    kc = kc + 1;
end

[mr mc] = size(mat);
fkr = floor(kr/2); % number of rows to copy on top and bottom
fkc = floor(kc/2); % number of columns to copy on either side
% this looks messy, but it's not.
% we use matlab's 2D convolution 
% (1) we have to expand the matrix (mat) by making copies of the border
%  rows and columns, plus the corner elements get special treatment.
% (2) we have to flip up-down, left right the kernel matrix (because we
%  are using conv2 (although ker will almost always be symmetric, so this
%  is not a big deal - but we do it because it is the right thing)
% (3) use the 'valid' flag (because we have expanded the original matrix
%  ourselves and do not want any zero-padding influence.
smat = conv2(...
    [mat(1,1)*ones(fkr,fkc) ones(fkr,1)*mat(1,:) mat(1,mc)*ones(fkr,fkc);
    mat(:,1)*ones(1,fkc) mat mat(:,mc)*ones(1,fkc)
    mat(mr,1)*ones(fkr,fkc) ones(fkr,1)*mat(mr,:) mat(mr,mc)*ones(fkr,fkc)],...
    flipud(fliplr(kmat)),'valid');


function peaks = findPeaks(tH,maxTH,aBin,dBin,distanceThresh)

% peaks = findPeaks(tH,maxTH,aBin,dBin,distanceThresh)
%
% This function searches the smoothed histogram created by 'DUETanalysis'
% for peak locations.  
%
% Input parameters
% --------------------
% tH(a,d) - thresholded, weighted and smoothed histogram of the number of
%           time-frequency frames with mixing parameters (alpha,delta)
%
% aBin - the bins for the histogram in the alpha dimension
%
% dBin - the bins for the histogram in the delta dimension
%
% distanceThresh - the Euclidean distance that all peaks must be from
%                  eachother.  since hist is from (-1,1) in both
%                  dimensions, start with values between 0.01 and 0.1
%
% Output values
% -----------------------
% peaks - an array of peak locations in the alpha and delta dimensions
%
% author - John Woodruff, May 11, 2005.


% find the number of bins used for alpha and delta (this should be equal to
% 'histRes'+1)
[numA,numD] = size(maxTH);

% find indexes (a,d) of all local maxima
i = 1;
for a=1:numA
    for d=1:numD
        if maxTH(a,d) > 0
            if i > 1
                
                % check distance from new potential peak to all other found
                % peaks
                for j=1:length(peaksA)
                    distance = ((aBin(a)-aBin(peaksA(j)))^2 + (dBin(d)-dBin(peaksD(j)))^2)^0.5;
                    
                    % if distance is too small, check which peak is higher,
                    % if the new one is higher than the old, replace the
                    % old, otherwise ignore new peak
                    if distance < distanceThresh
                        newPeak = tH(a,d);
                        oldPeak = tH(peaksA(j),peaksD(j));
                        if newPeak > oldPeak
                            peaksA(j) = a;
                            peaksD(j) = d;
                        end
                        break;                        
                    % if distance is not too small, add peak to the list
                    elseif j==length(peaksA)
                         peaksA(i) = a;
                         peaksD(i) = d;
                         i = i+1;
                    end
                end
                
            % add first peak found
            else
                peaksA(1) = a;
                peaksD(1) = d;
                i = i+1;
            end
        end
    end
end

% create an array of peaks locations 
peaks = [peaksA;peaksD];
