%   Degenerate Unmixing Estimation Technique (DUET) (from Rickard's work)
%       rickard_duet;
%
%   See also rafii_duet, boxplot_summary, loacl_peaks

%   Author: Zafar RAFII (zafarrafii@u.northwestern.edu)
%   Last update: October 2010

function rickard_duet

% ->
[file,rep] = uigetfile('*.wav','Select a stereo wav file to unmix');
if file ~= 0
    file = [rep,file];
else
    return
end
[x,fs] = audioread(file);
x1 = x(:,1);
x2 = x(:,2);
% <-

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. analyze the signals - STFT

wlen = 1024; timestep = 512; numfreq = 1024;
% ->
% wlen = 2048; timestep = 1024; numfreq = 2048;
% <-
awin = hamming(wlen);                                           % analysis window is a Hamming window

tf1 = tfanalysis(x1,awin,timestep,numfreq);                     % time-freq domain
tf2 = tfanalysis(x2,awin,timestep,numfreq);                     % time-freq domain
tf1(1,:) = []; tf2(1,:) = [];                                   % remove dc component from mixtures to avoid dividing by zero frequency in the delay estimation

% calculate pos/neg frequencies for later use in delay calc
freq = [(1:numfreq/2) ((-numfreq/2)+1:-1)]*(2*pi/(numfreq));
fmat = freq(ones(size(tf1,2),1),:)';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. calculate alpha and delta for each t-f point

R21 = (tf2+eps)./(tf1+eps);                                     % time-freq ratio of themixtures

%%% 2.1 HERE WE ESTIMATE THE RELATIVE ATTENUATION (alpha) %%%
a = abs(R21);                                                   % relative attenuation between the two mixtures
alpha = a-1./a;                                                 % 'alpha' (symmetric attenuation)

%%% 2.2 HERE WE ESTIMATE THE RELATIVE DELAY (delta) %%%%
delta = -imag(log(R21))./fmat;                                  % 'delta' relative delay

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3. calculate weighted histogram

p = 1; q = 0;                                                   % powers used to weight histogram
tfweight = (abs(tf1).*abs(tf2)).^p.*abs(fmat).^q;               % weights
maxa = 0.7; maxd = 3.6;                                         % hist boundaries
abins = 35; dbins = 50;                                         % number of hist bins
    
% only consider time-freq points yielding estimates in bounds
amask = (abs(alpha)<maxa)&(abs(delta)<maxd);
alpha_vec = alpha(amask);
delta_vec = delta(amask);
tfweight = tfweight(amask);

% determine histogram indices
alpha_ind = round(1+(abins-1)*(alpha_vec+maxa)/(2*maxa));
delta_ind = round(1+(dbins-1)*(delta_vec+maxd)/(2*maxd));

% FULL-SPARSE TRICK TO CREATE 2D WEIGHTED HISTOGRAM
A = full(sparse(alpha_ind,delta_ind,tfweight,abins,dbins));

% smooth the histogram - local average 3-by-3 neighboring bins
A = twoDsmooth(A,3);

% plot 2-D histogram
% mesh(linspace(-maxd,maxd,dbins),linspace(-maxa,maxa,abins),A);
% ->
figure,
mesh(linspace(-maxd,maxd,dbins),linspace(-maxa,maxa,abins),A);
pause
% <-

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4. peak centers (determined from histogram)

% numsources = 2;
% peakdelta = [0 0];
% peakalpha=[0 -0.4];
% ->
numsources = input('How many sources?\n');
peaks = local_peaks(A,[3,3]);                                   % Local extrema/peaks
peaks = flipud(sortrows(peaks,3));                              % Sort peaks by decreasing level value
peaks = peaks(1:numsources,1:2);                                % Keep only the indices of the first r peaks

peakalpha = peaks(:,1);
peakdelta = peaks(:,2);
peakalpha = 2*maxa*(peakalpha-1)/(abins-1)-maxa;
peakdelta = 2*maxd*(peakdelta-1)/(dbins-1)-maxd;
% <-
    
% convert alpha to a
peaka = (peakalpha+sqrt(peakalpha.^2+4))/2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 5. determine masks for separation

bestsofar = Inf*ones(size(tf1));
bestind = zeros(size(tf1));
for i=1:length(peakalpha)
    score = abs(peaka(i)*exp(-sqrt(-1)*fmat*peakdelta(i))...
    .*tf1-tf2).^2/(1+peaka(i)^2);
    mask = (score<bestsofar);
    bestind(mask) = i;
    bestsofar(mask) = score(mask);
end

% ->
figure,
imagesc(bestind);
pause
% <-

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 6.& 7. demix with ML alignment and convert to time domain

est = zeros(numsources,length(x1));                             % demixtures
for i=1:numsources
    mask = (bestind==i);
    esti = tfsynthesis([zeros(1,size(tf1,2));
    ((tf1+peaka(i)*exp(sqrt(-1)*fmat*peakdelta(i)).*tf2)...
    ./(1+peaka(i)^2)).*mask],...
    sqrt(2)*awin/wlen,timestep,numfreq);
    est(i,:)=esti(1:length(x1))';
    % add back into the demix a little bit of the mixture
    % as that eliminates most of the masking artifacts
%     soundsc(est(i,:)+0.05*x1,fs); pause;                        % play demixture
    % ->
    wavwrite(est(i,:)'+0.05*x1,fs,nbits,[file(1:end-4),'_',num2str(i),'.wav']);
    % <-
end
