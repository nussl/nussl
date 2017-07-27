% Inverse Short-Time Fourier Transform
%   x = istft(X,winfft,overlap);
%
% Input:
%   X: Short-Time Fourier Transform
%   winfft: analysis window (default: hamming(1024))
%   overlap: analysis overlap (default: round(length(winfft)/2))
%
% Output:
%   x: syntesized signal [length,#channels]

function x = istft(X,winfft,overlap)

if nargin < 2, winfft = hamming(1024); end
if nargin < 3, overlap = round(length(winfft)/2); end

[nfft,m,c] = size(X);                                           % nfft = analysis window length, m = #frames, c = #channels
X = ifft(X);                                                    % IFFT on the frames for all the channels
step = nfft-overlap;                                            % Analysis step
l = m*step+overlap;                                             % Synthesized length for x
I = (1:step:l-nfft+1);                                          % Vector of the overlapping indexes

if overlap < 2                                                  % Case overlap = 0 or 1
    ul = 0.5*ones(overlap);                                     % Left side of the unwindow
    ur = 0.5*ones(overlap);                                     % Right side of the unwindow
    unwin = [ul,ones(1,nfft-2*overlap),ur];                     % Overlapping unwindow
else
    ul = linspace(0,1,overlap);
    ur = linspace(1,0,overlap);
    if overlap <= 0.5*nfft                                      % Case 2 <= overlap <= 0.5*nfft
        unwin = [ul,ones(1,nfft-2*overlap),ur];                 % Overlapping unwindow
    else                                                        % Case overlap > 0.5*nfft
        v = step/(overlap-1);                                   % Value smaller than one
        unwin = [linspace(0,v,step+1),...
            v*ones(1,nfft-2*step-2),linspace(v,0,step+1)];
    end
end

o = ones(1,c);
winfft = winfft(:)+eps;
unwin = (unwin'./winfft)*o;                                     % Unwindow for the in-between frames for all the channels
unwin1 = ([ones(1,step),ur]'./winfft)*o;                        % Unwindow for the first frame for all the channels
unwinm = ([ul,ones(1,step)]'./winfft)*o;                        % Unwindow for the last frame for all the channels

x = zeros(l,c);
x(I(1):I(1)+nfft-1,:) = reshape(real(X(:,1,:)),[nfft,c,1]).*unwin1;
for j = 2:m-1
    x(I(j):I(j)+nfft-1,:) = x(I(j):I(j)+nfft-1,:)...
        +reshape(real(X(:,j,:)),[nfft,c,1]).*unwin;
end
x(I(m):I(m)+nfft-1,:) = x(I(m):I(m)+nfft-1,:)...
    +reshape(real(X(:,m,:)),[nfft,c,1]).*unwinm;
