%   REpeating Pattern Extraction Technique (REPET)
%       repet;
%
%   See also beat_spectrum, repet_period

%   Author: Zafar RAFII (zafarrafii@u.northwestern.edu)
%   Last update: February 2011

function repet

[filename,pathname] = uigetfile({'*.wav';'*.mp3'}, ...
    'Select *.wav or *.mp3.');
if isequal(filename,0)                                          % If 'Cancel', return
    return
end

file = [pathname,filename];                                     % File with path & extension
[~,filename,fileext] = fileparts(filename);                     % File name (without path & extension) & file extension
if strcmp(fileext,'.wav')
    [x,fs,nbits] = wavread(file);
elseif strcmp(fileext,'.mp3')
    [x,fs,nbits] = mp3read(file);
else
    error('Extension unknown.');
end

N = 2^nextpow2(fs*0.04);                                        % Analysis window length (next power of 2) (music signals ~ 40 msec)
winfft = hamming(N);
overlap = N/2;
X = stft(x,winfft,overlap);                                     % Short-Time Fourier Transform
V = abs(X(1:N/2+1,:,:));                                        % Magnitude spectrogram (including the DC component)
[n,m,l] = size(V);

b = beat_spectrum(mean(V.^2,3));                                % Beat spectrum of the mean power spectrogram
b = b/b(1);                                                     % Normalization
figure;
plot(b);
p = input('Repeating pattern period?\n');                       % Estimated period of the repeating structure (start at lag 0!)
% p = repet_period(b);

r = ceil(m/p);                                                  % Number of repeating segments (including the last one)
L = length(x);
x1 = zeros(L,l);
for k = 1:l                                                     % Loop on the channels
    V0 = V(:,:,k);                                              % Magnitude spectrogram of channel k
    V1 = [V0,nan(n,r*p-m)];                                     % Nan-padding to have an integer number of segments
    V1 = reshape(V1,[n*p,r]);                                   % Reshape such that the columns are the segments
    V1 = [median(V1(1:n*(m-(r-1)*p),1:r),2); ...                % Median of the parts repeating for all the r segments (including the last one)
        median(V1(n*(m-(r-1)*p)+1:n*p,1:r-1),2)];               % Median of the parts repeating only for the first r-1 segments (empty if m = r*p)
    V1 = reshape(repmat(V1,[1,r]),[n,r*p]);                     % Duplicate repeating segment model and reshape back to have [n x r*p]
    V1 = V1(:,1:m);                                             % Truncate to the original number of frames to have [n x m]
    
    [~,M] = max(cat(3,V0-V1,V1),[],3);                          % Mask built from the max between the mixture-model (~non-repeating) (=1) & the model (~repeating) (=2)
    M = M-1;                                                    % Binary time-frequency mask (0 = non-repeating & 1 = repeating)
    M = cat(1,M,flipud(M(2:end-1,:)));                          % Symmetrize the mask to have [N x m]
    M = istft(M.*X(:,:,k),winfft,overlap);                      % Estimated repeating signal
    x1(:,k) = M(1:L);                                           % Truncate to L if too long (because of the zero-padding in STFT)
end

file1 = fullfile(pathname,[filename,'_1',fileext]);
file2 = fullfile(pathname,[filename,'_2',fileext]);
if strcmp(fileext,'.wav')
    wavwrite(x1,fs,nbits,file1);
    wavwrite(x-x1,fs,nbits,file2);
elseif strcmp(fileext,'.mp3')
    mp3write(x1,fs,nbits,file1);
    mp3write(x-x1,fs,nbits,file2);
end
