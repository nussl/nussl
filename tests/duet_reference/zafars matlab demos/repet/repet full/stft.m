%   Short-Time Fourier Transform (using fft)
%       X = stft(x,winfft,overlap,opt);
%
%   Input(s):
%       x: sampled signal [#samples x #signals]
%       winfft: analysis window (default: hamming(1024))
%       overlap: analysis overlap (default: round(length(winfft)/2))
%       opt: if 0, process the signal in one time using 'buffer' (default)
%            else, process the signal one frame at a time (save memory)
%
%   Output(s):
%       X: Short-Time Fourier Transform [#bins x #frames x #signals]
%
%   See also istft, fft, buffer

%   Author: Zafar RAFII (zafarrafii@u.northwestern.edu)
%   Last update: October 2010

function X = stft(x,winfft,overlap,opt)

if nargin < 4, opt = 0; end
if nargin < 2, winfft = hamming(1024); end
if nargin < 3, overlap = round(length(winfft)/2); end

[l,p] = size(x);
nfft = length(winfft);                                          % Analysis window length
step = nfft-overlap;                                            % Analysis step
m = ceil((l-overlap)/step);                                     % Number of frames (with possible zero-padding)

X = zeros(nfft,m,p);
if opt == 0;
    for k = 1:p
        X(:,:,k) = buffer(x(:,k),nfft,overlap,'nodelay');       % Buffer the signal for each channel
        X(:,:,k) = X(:,:,k).*repmat(winfft(:),[1,m]);           % Windowing of the frames for each channel
        X(:,:,k) = fft(X(:,:,k));                               % fft of the frames for each channel
    end
    
else
    x = [x;zeros(m*step+overlap-l,p)];                          % Zero-padding
    for k = 1:p
        for j = 1:m
            X(:,j,k) = fft(x((1:nfft)+step*(j-1),k).*winfft);   % Windowing & fft of each frame for each the channel
        end
    end
end
