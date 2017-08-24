function [Sout,Fout,Tout,Pout]=f_stft(x,winlen,wintype,ovp,nfft,fs,Opt)

% This function computes the time-dependent FT (Spectrogram) of a signal.
% Inputs:
% x : vector containig samples of the time signal
% winlen : length of the window
% wintype: window type,string (rectangular,bartlett,hamming,hanning,blackman)
% ovp : # of overlapping samples
% nfft : # of frequency samples
% fs : sampling rate of the signal
% Opt: structure array containing optional inputs:
%      Fmax: (optional) maximum frequency in the spectrogram matrix and plot 
%            (FFT is calculated for the whole nfft points and then a portion 
%            of it is put into the spectrogram matrix 
%            By default the maximum frequency is equal to fs/2
%      s2: (optional) binary input indicating a spectrogram format 
%          (1 means two-sided, 0 means one-sided)
%           default is a single-sided spectrogram
%
%
% Outputs:
% Sout: short-time Fourier transform 
% Fout: frequency ventor
% Tout: time vector
% Pout: PSD of the signal
% Only spectrogram plot if no output is specified


%% Error messages

Lx=length(x);

if winlen>Lx 
    error('Lenghth of the window must be smaller than or equal the length of the sequence X');
elseif ovp>winlen
    error('# of overlapping samples must be smaller than or equal the length of the window');
elseif nfft<winlen
      error('# of frequency smaples must be more than the length of the window');
end

%% Check the number of inputs

if nargin==6
   Fmax=fs/2;
   s2=0;
   
elseif nargin==7 && ~isfield(Opt,'Fmax')
     Fmax=fs/2;
     s2=Opt.s2;
 
elseif nargin==7 && ~isfield(Opt,'s2')
    s2=0;    
    Fmax=Opt.Fmax;
    
else
    Fmax=Opt.Fmax;
    s2=Opt.s2;
end

%% Split data into blocks and zero pad:

hop=winlen-ovp; 

% zero-pad the vector at the beginning and end to reduce the window
% tapering effect

if ovp>=hop
    
    ovp_hop_ratio=ceil(ovp/hop);
    NumBlock=ceil(Lx/hop);
    Lxz=(ovp_hop_ratio+NumBlock)*hop+ovp;
    NumBlockZ=floor((Lxz-ovp)/hop);
    
    xz=zeros(Lxz,1);
    xz(ovp_hop_ratio*hop+(1:Lx))=x;
    
elseif ovp<hop
    
    NumBlock=ceil(Lx/hop);
    Lxz=(1+NumBlock)*hop+ovp; %winlen;
    NumBlockZ=floor((Lxz-ovp)/hop); %winlen)/hop);
    
    xz=zeros(Lxz,1);
    xz(hop+(1:Lx))=x;
        
end

% NumBlockZ=ceil((Lx-winlen)/hop);
% Lxz=NumBlockZ*hop+winlen;
% xz=zeros(Lxz,1);
% xz(1:Lx)=x;


%% Generate samples of a normalized window:

switch wintype
    case 'rectangular'
        W=ones(winlen,1);
    case 'bartlett'
        W=bartlett(winlen);
    case 'hamming'
        W=hamming(winlen);
    case 'hanning'
        W=hann(winlen);
    case 'blackman'
        W=blackman(winlen);
end
Wnorm2=(norm(W))^2;


%% Generate freuqency vectors 

%nfft=2^nextpow2(nfft);
nfft=nfft+mod(nfft,2);
F=(fs/2)*linspace(0,1,nfft/2+1);
%fr=fs/nfft;
%Fel_max=find(abs(F-Fmax)<fr);
%Fel_max=Fel_max(1);
Fel_max=nfft/2+1;
F=F(1:Fel_max);


%% Take the fft of each block

SFmax=zeros(Fel_max,NumBlockZ);
PFmax=zeros(Fel_max,NumBlockZ);

for i=0:NumBlockZ -1
    xw = W.* xz((1+i*hop):(i*hop+winlen));    
    XX=fft(xw,nfft);
    XX_trun=XX(1:Fel_max);
    
    SFmax(:,i+1)=XX_trun;
    PFmax(:,i+1)=(1/fs)*((abs(SFmax(:,i+1)).^2)/Wnorm2);   
end

Th=hop/fs;
T=0:Th:(NumBlockZ-1)*Th;


%% Form the two-sided spectrogram if indicated by s2 

if s2==1
    
 SFmax_conj=conj(SFmax(end-1:-1:2,:)); 
  
 SFmax=[SFmax_conj;SFmax];
 
 PFmax=[PFmax(end-1:-1:2,:);PFmax];
 
 F=[-F(end-1:-1:2),F];
 
end

%% Plotting the Spectrogram and returning the results

if nargout==1
Sout=SFmax;

elseif nargout==3

Sout=SFmax;
Tout=T;
Fout=F;
    
elseif nargout==4
Pout=PFmax;
Sout=SFmax;
Tout=T;
Fout=F;

elseif nargout==0
        
% Plot    
    
SP=10*log10(abs(PFmax));

set(0,'defaultfigurecolor','w');
set(0,'defaultaxesfontsize',14,'defaulttextfontsize',14);

imagesc(T,F,SP);
set(gca,'YDir','normal');
colorbar;
xlabel('Time(sec)');
ylabel('Frequency (Hz)');
title(['Spectrogram - Window Type: ',wintype]);

end











