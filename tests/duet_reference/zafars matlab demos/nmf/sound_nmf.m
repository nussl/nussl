% Synthesis of the components separated by NMF
%   xxk = sound_nmf(name,W,H,X,winfft,overlap,fs,nbits,fusion,T,option);
%
% Input:
%   W: basis matrix
%   H: weight matrix
%   X: full complex spectrum
%   winfft: analysis window (default: winfft=hamming(1024))
%   overlap: analysis overlap (default: half overlapping)
%   fs: Sampling frequency (default: fs=44100 Hz)
%   nbits: Number of bits per sample (default: nbits=16 bits)
%   name: name for the syntesized wav sounds (without the ".wav" extension)
%   fusion: =0: no fusion (default)
%           =1: fusion of the similar components from an unique measures table
%           =2: fusion from the computation of new measures table after each double fusion
%           =3: fusion from an overlapping table
%   T: threshold for the fusion or Fusion vector if fusion==0 && T~=0 (default: T=0)
%   option: =0: write the signals without saving them (default)
%           =1: save the signals without writing them
%           =2: save & write the signals
%
% Output:

function xxk = sound_nmf(W,H,X,winfft,overlap,fs,nbits,name,fusion,T,option)

if nargin<11, option=0; end
if nargin<10, T=0; end
if nargin<9, fusion=0; end
if nargin<8, name = 'component'; end
if nargin<7, nbits=16; end
if nargin<6, fs = 16000; end
if nargin<5, overlap = size(X,1)/2; end
if nargin<4, winfft = hamming(1024); end

% Parameters:

[r,m] = size(H);
E = sum(H,2)/sum(H(:));                                         %Vector of the normalized energy of the components for the computation of the weighted continuous part
o = ones(1,m);                                                  %Vector of ones to form the weighted continuous part
V = W*H+eps;                                                    %Sum of the spectrograms of all the components

% Fusion indexes:

if fusion==0 && T==0                                            %No fusion
    K = 1:r;                                                    %Vector of the indexes of the components
    q = r;                                                      %Number of components
    disp([num2str(q),' components without fusion:']);
elseif fusion==0 && T~=0                                        %Fusion with a given vector of fusionned indexes
    K = T;
    q = length(unique(K));                                      %Number of different components
    disp([num2str(q),' components with forced fusion:']);
else                                                            %Fusion from a vector of fusionned indexes
    [K,q] = fusion_nmf(W,H,fs,fusion,T);
    disp([num2str(q),' components with fusion:']);
end

% Signal synthesis:

q = 0;
xxk = [];
for k=1:r                                                       %Loop on all the indexes
    f = find(K==k);                                             %Indexes vector of the similar components
    if ~isempty(f)                                              %If the index exists
        q = q+1;
        Vk = (W(:,f)*H(f,:))./V;                                %The similar spectra are fusionned and divided by the total sum of the spectra
        W(:,f) = []; H(f,:) = []; K(f) = [];                    %Free memory
        Vk = [sum(E(f))*o;Vk;flipud(Vk(1:end-1,:))];            %#ok<AGROW> %Synthesis of the full fusionned spectrum (weighted continous part & spectrum & flipud spectrum)
        Vk = Vk.*X;                                             %Synthesis of the continuous part and the phase: the final total sum is conservative!
        xk = istft(Vk,winfft,overlap); clear Vk                  %Synthesis of the fusionned signals from their spectrum
        
        if option==0                                            %Write the signals without saving them
            name_q = [name,'_',num2str(q)];
            wavwrite(xk,fs,nbits,name_q);
            disp([name,' ',num2str(q),' written']);
        elseif option==1                                        %Save the signal without writing them
            xxk = [xxk,xk]; %#ok<AGROW>
            disp([name,' ',num2str(q),' saved']);
        elseif option==2                                        %Save & write the signals
            name_q = sprintf('%s_%d.wav',name,q);
            xxk = [xxk,xk]; %#ok<AGROW>
            wavwrite(xk,fs,nbits,name_q);
            disp([name,' ',num2str(q),' saved & written']);
        end
        clear xk;
    end
    
end
disp('')
