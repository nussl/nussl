function [sources,P,Q] = PROJETanechoic(filename,J,M,L,D,T,R,cf)
%PROJET Stereo Sound Source Separation via PROJection Estimation Technique
%This code is a MATLAB implementation of the projection-based spatial audio
%separation algorithm presented in "Projection-based demixing of spatial audio",
%published in IEEE TASLP.
%
%If you use this code, please reference the following paper:
%@article{fitzgeraldPROJETb,
%  TITLE = {{Projection-based demixing of spatial audio}},
%  AUTHOR = {D. Fitzgerald and A. Liutkus and R. Badeau},
%  JOURNAL = {{IEEE Transactions on Audio, Speech and Language Processing}},
%  PUBLISHER = {{Institute of Electrical and Electronics Engineers}},
%  YEAR = {2016},
%  MONTH = May,
%}
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
%
%
%   [sources,P,Q] = PROJETanechoic(filename,ns,npan,nproj)
%   INPUTS:
%   filename - name of stereo wav file to be separated e.g. 'test.wav'
%   J - number of sources to separate
%   M - number of pan directions to project the stereo mixture on (default
%   = 10)
%   L - number of pan directions used in the signal model (default = 20)
%   D  - delay step size(default = 1 sample) - Can be non-integer
%   T - number of delay directions to project the stereo signal on
%   (default=10)
%   R - number of delay directions used in the signal model (default =20)
%   cf - indicates which cost function to use when separating the sources.
%          
%       1 uses heuristic NMF-type updates assuming the source spectrograms are 
%       distributed according to the multivariate Cauchy distribution.
%       2 uses the generalised Kullback Leibler divergence as a cost function
%
%       default = 2 
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

if nargin<8
    cf=4;
end
if nargin<7
    R=20;
end
if nargin<6
    T=10;
end
if nargin<5
    D=1;
end
if nargin<4
    L=20;
end
if nargin<3
    M=10;
end

%algorithm parameters
niter = 200; %number of iterations 
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

X(:,:,1)=yL+eps;
[yR]=stft(wav(:,2),nfft,nfft,hopsize);
X(:,:,2)=yR+eps;
fprintf('Projecting signal onto projection directions \n');
%initialise projection and panning directions
pan_dir=0:pi/(2*(L-1)):pi/2;
proj_dir=0:pi/(2*(M-1)):pi/2;


%set up model delays
R2=floor(R/2);
R2U=ceil(R/2);
shiftsteppan=(-R2*D):D:((R2U-1)*D);
nshpan=length(shiftsteppan);
SPanten=zeros(nshpan*L,Nf,2);


ind=1;
%create pan and delay model matrix
for l=1:R
    thetatemp=ones(Nf,1);
    SPanten(ind:(ind+R-1),:,1)= (thetatemp*cos(pan_dir))';
    thetatemp=exp(-1i*2*pi*(0:Nf-1)*shiftsteppan(l)/nfft)';
    SPanten(ind:(ind+R-1),:,2)=(thetatemp*sin(pan_dir))';
    ind=ind+R;
end

%set up projection delays
T2=floor(T/2);
T2U=ceil(T/2);
shiftstepproj=(-T2*D):D:((T2U-1)*D);
nshproj=length(shiftstepproj);
SProjten=zeros(nshproj*T,Nf,2);  
ind=1;
%create pan and delay projection matrix
for l=1:T
    thetatemp=ones(Nf,1);
    SProjten(ind:(ind+T-1),:,1)= (thetatemp*cos(proj_dir-(pi/2)))';
    thetatemp=exp(1i*2*pi*(0:Nf-1)*shiftstepproj(l)/nfft)';
    SProjten(ind:(ind+T-1),:,2)=(thetatemp*(sin(proj_dir-(pi/2))))';
    ind=ind+T; 

end
SProjten=SProjten+eps;

C=zeros(Nf,Nt,T*M);
Ccomp=C;
%generate projections
for m=1:(T*M)
    curproj=squeeze(SProjten(m,:,:))+eps;

    projrep=repmat(curproj,[1 1 Nt]);
    projrep=permute(projrep,[1 3 2]);
    Cc=sum((projrep.*X),3);
    Ccomp(:,:,m)=Cc;
    C(:,:,m)=abs(Cc);
end
C=C+eps;
P=abs(randn(Nf,Nt,J))+1;
Q=abs(randn(J,R*L))+1;
fprintf('Learning model parameters \n');
if cf==1 %Cauchy Heuristic updates
    C2=C.^2;
    for n=1:niter
        uQf=zeros([size(Q) Nf]);
        lQf=uQf;
        %calculate updates for each frequency bin, and accumulate for
        %updating Q
        for p=1:Nf
            curpan=(squeeze(SPanten(:,p,:)));
            curproj=(squeeze(SProjten(:,p,:)));
            K=curpan*curproj.' +eps;
            K=abs(K)+eps;
            QK=Q*K;
            est=(squeeze(P(p,:,:)))*QK +eps;

            e2=est.^2;
            ei=(1./(est+eps));
            z=3*est./(squeeze(C2(p,:,:)) + e2);
    %       %update for P
            uP=ei*QK';
            lP=z*QK'+eps;
            P(p,:,:)=squeeze(P(p,:,:)).*(uP./lP);

            est=(squeeze(P(p,:,:)))*QK +eps;

            e2=est.^2;
            ei=(1./(est+eps));
            z=3*est./(squeeze(C2(p,:,:)) + e2);
%           Accumulate update for Q
            uQf(:,:,p)=(squeeze(P(p,:,:))'*ei)*K';
            lQf(:,:,p)=(squeeze(P(p,:,:))'*z)*K';

        end
        %update for Q
        lQ=sum(lQf,3)+eps;
        uQ=sum(uQf,3)+eps;
        Q=Q.*(uQ./lQ);
        fprintf('Completed iteration %d / %d \n',n,niter);
    end    
else
    for n=1:niter
        uQf=zeros([size(Q) Nf]);
        lQf=uQf;
        %calculate updates for each frequency bin, and accumulate for
        %updating Q
        for p=1:Nf
            curpan=(squeeze(SPanten(:,p,:)));
            curproj=(squeeze(SProjten(:,p,:)));
            K=curpan*curproj.' +eps;
            K=abs(K)+eps;
            QK=Q*K;
            est=(squeeze(P(p,:,:)))*QK +eps;
            ei=squeeze(C(p,:,:))./(est+eps);
    %       %update for P
            uP=ei*QK';
            lP=repmat(sum(QK,2)',Nt,1) +eps;
            P(p,:,:)=squeeze(P(p,:,:)).*(uP./lP);
            %Accumulate update for Q
            est=(squeeze(P(p,:,:)))*QK +eps;
            ei=squeeze(C(p,:,:))./(est+eps);
            uQf(:,:,p)=(squeeze(P(p,:,:))'*ei)*K';
            lQf(:,:,p)=(repmat(sum(squeeze(P(p,:,:)),1)',1,M*T))*K';

        end
        %update for Q
        lQ=sum(lQf,3)+eps;
        uQ=sum(uQf,3)+eps;
        Q=Q.*(uQ./lQ);
        fprintf('Completed iteration %d / %d \n',n,niter);
    end    

end

fprintf('Resynthesising separated sources by projection back to stereo. \n');
for m=1:J
    ys=zeros(Nf,Nt,2);
    for n=1:Nf
        curpan=(squeeze(SPanten(:,n,:)))+eps;
        curproj=(squeeze(SProjten(:,n,:)))+eps;
        K=curpan*curproj.';
        K=abs(K)+eps;
        QK=Q*K;
        cur=squeeze(P(n,:,:))+eps;
        est=cur*QK; 
        ccur=squeeze(Ccomp(n,:,:));
        ck=ccur.*(cur(:,m)*Q(m,:)*K)./(est+eps);
        yk=ck/curproj.';
        ys(n,:,:)=yk;
    end
    sources(m).d(:,1)=istftwin(squeeze(ys(:,:,1)),nfft,nfft,hopsize)./sc;
    sources(m).d(:,2)=istftwin(squeeze(ys(:,:,2)),nfft,nfft,hopsize)./sc;

end 

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


