%   Degenerate Unmixing Estimation Technique (DUET) with Constant Q Transform (CQT)
%       rafii_duet(opt);
%
%   Input(s):
%       opt: option for the type of time-frequency representation to use
%            0: Short-Time Fourier Transform (default)
%            1: Short-Time constant Q Transform
%       bound: option for the boundaries of the 2d histogram
%              0: fixed boundaries (default)
%              1: adaptive boundaries using boxplot_summary
%
%   See also rickard_duet, boxplot_summary, local_peaks

%   Author: Zafar RAFII (zafarrafii@u.northwestern.edu)
%   Last update: October 2010

function rafii_duet(opt,bound)

if nargin < 2, bound = 0; end
if nargin < 1, opt = 0; end

if (opt ~= 0) && (opt ~= 1)
    error('The first argument is either 0 (STFT) or 1 (STQT).')
end
if (bound ~= 0) && (bound ~= 1)
    error('The second argument is either 0 (fixed boundaries) or 1 (adaptive boundaries).')
end

[file,rep] = uigetfile('*.wav','Select a stereo wav file to unmix');
if file ~= 0
    file = [rep,file];
else
    return
end

display('Analyzing stereo mixture ...')                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[x,fs,nbits] = wavread(file);
if size(x,2) ~= 2
    error('The selected wav file to unmix needs to be stereo.')
end

if fs ~= 44100
    x = resample(x,44100,fs);                                   % Resample x at 44100 Hz (because of the preprocessed K)
    fs = 44100;
end
N = 2048;                                                       % Analysis window length (music signals <=> 40 msec)

switch opt
    case 0                                                      % DUET with STFT
        X = stft(x,hamming(N),N/2);
        [N,m,~] = size(X);
        n = floor(N/2);                                         % Number of frequency bins for the STFT spectrogram
        X1 = X(2:n+1,:,1);                                      % Left complex spectrogram
        X2 = X(2:n+1,:,2);                                      % Right complex spectrogram
        F = repmat(2*pi*(1:n)'/N,1,m);                          % Angular frequency matrix (w = 2*pi*f) to get time in samples (no /fs)
    case 1                                                      % DUET with STQT
        load K
        X = N*stqt(x,K,2048,1024);                              % Short Time constant Q Transform (default parameters) (unnormalized to be comparable with the unnormalized STFT)
        [n,m,~] = size(X);
        X1 = X(:,:,1);                                          % Left complex CQT spectrogram
        X2 = X(:,:,2);                                          % Right complex CQT spectrogram
        b = 24;                                                 % Number of bins per octave in the CQT
        fmin = note2freq('E0');                                 % Minimal frequency in the CQT
        F = repmat(2*pi*(fmin/fs)*2.^((0:n-1)'/b),1,m);         % Angular frequency matrix (w = 2*pi*f) to get time in samples (fmin/fs)
end

display('Building 2d histogram ...')                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

R = (X2+eps)./(X1+eps);
Alpha = abs(R);                                                 % Amplitude ratio matrix (no unity) (center in 1)
Alpha = Alpha - 1./Alpha;                                       % Symmetric amplitude ratio (centered at 0)
Delta = angle(R);                                               % angle = arg = imag(log(.)) (in rad [-pi,pi]) 
Delta = -Delta./F;                                              % Time difference matrix (no unity) (center in 0)

switch bound
    case 0                                                      % Fixed boundaries
        Amax = 0.7;
        Dmax = 3.6;
        M = (abs(Alpha)<Amax)&(abs(Delta)<Dmax);                % Boundaries mask
    case 1                                                      % Adaptive boundaries
        bp = boxplot_summary(Alpha);                            % Analysis of the distribution of the values in Alpha
        Amin = bp.lower_whisker;                                % Lowest datum still within 1.5 IQR of the lower quartile
        Amax = bp.upper_whisker;                                % Highest datum still within 1.5 IQR of the upper quartile
        bp = boxplot_summary(Delta);                            % Analysis of the distribution of the values in Delta
        Dmin = bp.lower_whisker;                                % Lowest datum still within 1.5 IQR of the lower quartile
        Dmax = bp.upper_whisker;                                % Highest datum still within 1.5 IQR of the upper quartile
        
        M = (Alpha>Amin)&(Alpha<Amax)&(Delta>Dmin)&(Delta<Dmax);% Boundaries mask (1=within-bounds & 0=out-of-bounds)
        Amax = max(abs(Amax),abs(Amin));
        Dmax = max(abs(Dmax),abs(Dmin));
end

Alpha = Alpha(M);
Delta = Delta(M);
Alpha = (Alpha/Amax+1)/2;                                       % Normalization to get the values between 0 & 1
Delta = (Delta/Dmax+1)/2;                                       % Normalization to get the values between 0 & 1

A_bins = 35;                                                    % Number of bins for the Alpha-histogram
D_bins = 50;                                                    % Number of bins for the Delta-histogram
Alpha = round(Alpha*(A_bins-1)+1);                              % Rounded values between 1 & Alpha_bins (avoiding 0 indices)
Delta = round(Delta*(D_bins-1)+1);                              % Rounded values between 1 & Delta_bins (avoiding 0 indices)

w1 = 1;                                                         % Weight on the magnitude spectrogram
w2 = 0;                                                         % Weight on the angular frequencies
W = ((abs(X1).*abs(X2)).^w1).*(F.^w2);                          % Weights for the histogram
W = W(M);

H = accumarray([Alpha,Delta],W,[A_bins,D_bins]);                % 2d histogram
H = twoDsmooth(H,3);                                            % Rickard's 2d smooth function

figure,
mesh(linspace(-Dmax,Dmax,D_bins),linspace(-Amax,Amax,A_bins),H)
pause

r = input('How many sources?\n');
peaks = local_peaks(H,[3,3]);                                   % Local extrema/peaks
peaks = flipud(sortrows(peaks,3));                              % Sort peaks by decreasing level value
peaks = peaks(1:r,1:2);                                         % Keep only the indices of the first r peaks

display('Forming binary masks ...')                             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A_peaks = peaks(:,1);                                           % Alpha-coordinates of the peaks
D_peaks = peaks(:,2);                                           % Delta-coordinates of the peaks
A_peaks = 2*Amax*(A_peaks-1)/(A_bins-1)-Amax;                   % Denormalization from [1,A_bins] to [-Amax,Amax]
A_peaks = (A_peaks+sqrt(A_peaks.^2+4))/2;                       % Symmetric amplitude ratio to amplitue ratio
D_peaks = 2*Dmax*(D_peaks-1)/(D_bins-1)-Dmax;                   % Denormalization from [1,D_bins] to [-Dmax,Dmax]

if opt == 1                                                     % If analysis via STQT, synthesis via STFT
    X = stft(x,hamming(N),N/2);
    [N,m,~] = size(X);
    n = floor(N/2);
    X1 = X(2:n+1,:,1);
    X2 = X(2:n+1,:,2);
    F = repmat(2*pi*(1:n)'/N,1,m);
end
    
Masks = zeros(n,m);                                             % Binary masks
Temp = +Inf*ones(n,m);                                          % Matrix for comparison
for k = 1:r                                                     % Loop on the estimated sources
    Maxlike = (abs(A_peaks(k)*exp(-1i*F*D_peaks(k)).*X1-X2).^2)...
        /(1+A_peaks(k)^2);                                      % Maximum likelihood
    ind = (Maxlike<Temp);                                       % Indices of the closest peaks location (so far)
    Masks(ind) = k;
    Temp(ind) = Maxlike(ind);                                   % Update matrix for comparison
end

figure,
imagesc(Masks);
pause
    
display('Estimating sources ...')                               %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X = mean(X,3);                                                  % Mean STFT over the channels
VV = zeros(n,m,r);                                              % Estimated spectrograms of the sources
for k = 1:r                                                     % Loop on the estimated sources
    Mask = (Masks==k);                                          % Binary mask for source k
    Y = ((X1+A_peaks(k)*exp(1i*F*D_peaks(k)).*X2)...
        ./(1+A_peaks(k)^2));                                    % Maximum likelihood
    VV(:,:,k) = Mask.*Y;                                        % Estimated spectrograms for the sources
end
XX = stft_wiener_filtering(X,abs(VV));                          % Wiener filtering

display('Synthesizing sources ...')                             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

file = file(1:end-4);                                           % Name without the '.wav'
l = length(x);
for k = 1:r
    xk = istft(XX(:,:,k),hamming(N),N/2);                       % Inverse STFT of source k
    xk = xk(1:l);                                               % Truncate the length to the original length
    wavwrite(xk,fs,nbits,[file,'_',num2str(k),'.wav']);
    display(['  -> source ',num2str(k),' written'])
end
