%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. analyze the signals - STFT

wlen = 1024; timestep = 512; numfreq = 1024;
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

p = 0; q = 0;                                                   % powers used to weight histogram
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
mesh(linspace(-maxd,maxd,dbins),linspace(-maxa,maxa,abins),A);
