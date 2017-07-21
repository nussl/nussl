function [sources,P,Q] = PROJETinstantaneous(filename,J,M,L,cf)
%PROJET Stereo Sound Source Separation via PROJection Estimation Technique
%This code is a MATLAB implementation of the projection-based spatial audio
%separation algorithm presented in "PROJET - Spatial Audio Separation using Projections",
%presented at ICASSP 2016.
%
%If you use this code, please reference the following paper:
% @inproceedings{fitzgerald:hal-01248014,
%    TITLE = {{PROJET - Spatial Audio Separation Using Projections}},
%    AUTHOR = {D. Fitzgerald and A. Liutkus and R. Badeau},
%    BOOKTITLE = {{41st International Conference on Acoustics, Speech and Signal Processing (ICASSP)}},
%    ADDRESS = {Shanghai, China},
%    PUBLISHER = {{IEEE}},
%    YEAR = {2016},
% }
%------------------------------------------------------------------------------
%Redistribution and use in source and binary forms, with or without
%modification, are permitted provided that the following conditions are met:
%    * Redistributions of source code must retain the above copyright
%      notice, this list of conditions and the following disclaimer.
%    * Redistributions in binary form must reproduce the above copyright
%      notice, this list of conditions and the following disclaimer in the
%      documentation and/or other materials provided with the distribution.
%    * Neither the name of the <organization> nor the
%      names of its contributors may be used to endorse or promote products
%      derived from this software without specific prior written permission.
%
%THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
%ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
%WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
%DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
%DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
%(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
%LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
%ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
%(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
%SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%------------------------------------------------------------------------------
%   [sources,P,Q] = PROJETinstantaneous(filename,ns,npan,nproj)
%   INPUTS:
%   filename - name of stereo wav file to be separated e.g. 'test.wav'
%   J - number of sources to separate
%   M - number of directions to project the stereo mixture on (default
%   = 10)
%   L - number of pan directions used in the signal model (default = 30)
%   cf - indicates which cost function to use when separating the sources.
%        
%       1 uses majorization-equalisation type updates, assuming a
%       multivariate Cauchy distribution for the source spectrograms
%       2 uses the generalised Kullback Leibler divergence as a cost function
%       3 uses the Itakura-Saito divergence as a cost
%       function
%       4 uses heuristic NMF-type updates assuming the source spectrograms are 
%       distributed according to the multivariate Cauchy distribution.
%       default = 4 
%   
%   
%   OUTPUTS:
%   sources - structure containing the J stereo sources e.g. sources(1).d
%   contains the waveform of the 1st separated source.
%   P - J x Nf x Nt tensor containing the (single channel)spectrograms of
%   the separated sources
%   Q - panning matrix, which indicates which panning directions are active
%   with a given source
%
%   EXAMPLE USAGE:
%   [sources,P,Q]=PROJET('test.wav',5,10,30,'KL');
%
%Copyright Derry FitzGerald, 2016

if nargin<5
    cf=4;
end
if nargin<4
    L=30;
end
if nargin<3
    M=10;
end

%algorithm parameters
niter = 500; %number of iterations - better quality at niter = 1000
nfft = 4096; %size of fft to use
hopsize = 1024;  %hopsize between frames of the stft
sc=1.5;

%Read in wav file
fprintf('Loading file \n');
wav=wavread(filename);
%Get STFT for each channel
fprintf('Peforming STFT on each channel \n');
[yL]=stft(wav(:,1),nfft,nfft,hopsize);
[Nf,Nt]=size(yL);
%Note: vectorising the spectrograms for speed of computation
X=zeros(Nf*Nt,2);
X(:,1)=reshape(yL,Nf*Nt,1);
[yR]=stft(wav(:,2),nfft,nfft,hopsize);
X(:,2)=reshape(yR,Nf*Nt,1);
fprintf('Projecting signal onto projection directions \n');
%initialise projection and panning directions
pan_dir=0:pi/(2*(L-1)):pi/2;
proj_dir=0:pi/(2*(M-1)):pi/2;
%calculate panning matrix
panmat=[cos(pan_dir); sin(pan_dir)];
%calculate projection matrix
projmat=[sin(proj_dir); -cos(proj_dir)];

K=abs(panmat'*projmat);

%project stereo signal in directions set by panmat
Cc=X*projmat;
C=abs(Cc);
C2=C.^2;

%initialise PROJET model parameters
Q=abs(randn(J,L))+1;
%again vectorising source spectrograms for speed
P=abs(randn(Nf*Nt,J))+1;

%iterate updates of model parameters
fprintf('Learning model parameters \n');
for n=1:niter
    
    %update for P
    if cf==1 %Generalised Kullback-Leibler
        QK=Q*K;
        est=P*QK;
        ei=C./(est+eps);
        
        %update for P
        uP=ei*QK';
        lP=repmat(sum(QK,2)',Nf*Nt,1);
        P=P.*(uP./lP);
        
        est=P*(QK);
        ei=C./(est+eps);
        
        %update for Q
        uQ=(P'*ei)*K';
        lQ=(repmat(sum(P,1)',1,M))*K' +eps;
        Q=Q.*(uQ./lQ);
        
    elseif cf==2 %Cauchy Majorisation Equalisation
        QK=Q*K;
        est=P*QK;
        e2=est.^2;
        ei=(1./(est+eps));
        
        %update for P
        beta=(ei*QK');
        alpha=0.75*(est./(e2+C2+eps))*QK';
        P=(P.*beta)./(eps+alpha+(alpha.^2+2*beta.*alpha).^0.5);
        
        est=P*(QK);
        e2=est.^2;
        ei=(1./(est+eps));
        
        %update for Q
        beta=(P'*ei)*K';
        alpha=0.75*(P'*(est./(e2+C2+eps)))*K';
        Q=(Q.*beta)./(alpha+(alpha.^2+2*beta.*alpha).^0.5);
        
    elseif cf==3 %Itakuro-Saito Divergence
        QK=Q*K;
        est=P*QK;
        e2=est.^2;
        ei=C2./(e2+eps);
        z=1./(est+eps);
        
        %update for P
        uP=ei*QK';
        lP=z*QK'+eps;
        P=P.*(uP./lP);
        
        est=P*(QK);
        e2=est.^2;
        ei=C2./(e2+eps);
        z=1./(est+eps);
        
        %update for Q
        uQ=(P'*ei)*K';
        lQ=(P'*z)*K' +eps;
        Q=Q.*(uQ./lQ);
        
    else %Cauchy Heuristic
        QK=Q*K;
        est=P*QK;
        e2=est.^2;
        ei=(1./(est+eps));
        z=3*est./(C2 + e2+eps);
        
        %update for P
        uP=ei*QK';
        lP=z*QK'+eps;
        P=P.*(uP./lP);
        
        est=P*(QK);
        e2=est.^2;
        ei=(1./(est+eps));
        z=3*est./(C2 + e2+eps);
        
        %update for Q;
        uQ=(K*(ei'*P))';
        lQ=(K*(z'*P))';
        Q=Q.*(uQ./(lQ+eps));
        
    end
    fprintf('Completed iteration %d / %d \n',n,niter);
end

%recover source images by least squares projection back to stereo case
fprintf('Recovering sources \n');
est=P*(Q*K);
for j=1:J
    cj=Cc.*(P(:,j)*Q(j,:)*K)./(est+eps);
    yj=cj/projmat;
    yj=reshape(yj,Nf,Nt,2);
    sources(j).d(:,1)=istftwin(squeeze(yj(:,:,1)),nfft,nfft,hopsize)./sc;
    sources(j).d(:,2)=istftwin(squeeze(yj(:,:,2)),nfft,nfft,hopsize)./sc;
    
end
%reshape source specgtrograms to matrix form
Pb=zeros(J,Nf,Nt);
for j=1:J
    Pb(j,:,:)=reshape(P(:,j),Nf,Nt);
end
P=Pb;



end

function y = stft(wav,nfft, winsize, hopsize)

len = length(wav);
win = 0.5*(1-cos([0:(winsize-1)]/winsize*2*pi))';
nframes = 1 + floor((len - winsize)/hopsize);
y = zeros(1 + nfft/2, nframes);
nzb = floor( (nfft-winsize)/2 );
nza = nfft-winsize - nzb;

for nf = 1:nframes
   f = wav((nf-1)*hopsize + [1:winsize]);
   f=f.*win;
  if nfft > winsize
    fw = [zeros(1,nzb),f,zeros(1,nza)];
  end
  if nfft < winsize
    f = f(-nzb+(1:nfft));
  end
  F = fft(fftshift(f));
  % Keep bottom half of frame
  y(:,nf) = F(1:(1 + nfft/2))';%*norm;
end;

end

function sig =istftwin(y,nfft,winsize,hopsize)

window = 0.5*(1-cos([0:(winsize-1)]/winsize*2*pi));
[nr,nc]=size(y);
ofact=round(winsize/hopsize);
if ofact==2
    window=boxcar(nfft);
end

nsamps=(nc+ofact-1)*hopsize;
sig=zeros(nsamps,1);
pos=1;
nzb = floor( (nfft-winsize)/2 );
ya=zeros(1,nfft);
for i=1:nc  
   ya(1:nr)=y(1:nr,i);
   ya(nr+1:nfft)=conj(y(nr-1:-1:2,i));
   x=ifftshift(ifft(ya'));
   sig(pos:(pos+winsize-1))=sig(pos:(pos+winsize-1))+ x(nzb+1:(nzb+winsize)).*window';
   pos=pos+hopsize;
end
sig=real(sig);

end


