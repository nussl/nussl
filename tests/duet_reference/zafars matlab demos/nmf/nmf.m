% Non-negative Matrix Factorization (NMF)
%   [W,H,E] = nmf(V,W,H,costf,varmin,maxiter,alpha,beta,gamma,flag);
%
% Input:
%   V: matrix to factorize V ~= W*H  [n x m]
%   W: initial basis matrix [n x r]
%   H: initial weight matrix [r x m]
%   costf: cost function for the approximation error:
%         ='SED': squared Euclidean distance (= Frobenius Norm)
%         ='KLD': generalized Kullback-Leibler divergence (default)
%         ='ISD': Itakura-Saito divergence
%         ='CKL': KLD + smoothness, sparseness & orthogonality constraints
%   varmin: minimal relative variation of the approximation error (default: varmin=1e-3)
%   maxiter: maximal number of iterations (default: maxiter=50)
%   alpha: temporal continuity (smoothness) parameter
%   beta: sparseness parameter
%   gamma: orthogonality parameter
%   flag: =1 displays the iterations & approximation errors (default)
%
% Output:
%   W: final basis matrix
%   H: final weight matrix
%   E: approximation error vector

function [W,H,E] = nmf(V,W,H,costf,varmin,maxiter,alpha,beta,gamma,flag)

if nargin<10, flag = 1; end
if nargin<7, alpha = 0; beta = 0; gamma = 0; end
if nargin<6, maxiter = 100; end
if nargin<5, varmin = 1e-3; end
if nargin<4, costf = 'KLD'; end

[n,m] = size(V);
r = size(W,2);                                                  %Number of components
v = sum(V(:));                                                  %Total energy of V
eps = 2.2204e-16;
t = 0;                                                          %Initialized counter

if strcmp(costf,'SED')                                          %NMF with the squared Euclidean distance
    w = sum(W,1); W = W*diag(1./w);                             %Normalization of the components in W
    H = diag(w)*H; H = H*v/sum(H(:));                           %Transfer of the energy of the components of W in H
    WH = W*H;                                                   %The total energy of W*H has been forced to be equal to v
    e0 = (V-WH)/v; e0 = sum(e0(:).^2);                          %Initial approximation error (iteration 0)
    
    while(t < maxiter)
        
        % Update of W:
        W = W.*(V*H')./(WH*H'+eps); clear WH
        w = sum(W,1); W = W*diag(1./w);                         %Normalization of the components in W

        % Update of H:
        H = diag(w)*H;                                          %Transfer of the energy in H
        H = H.*(W'*V)./(W'*W*H+eps);
        
        % Approximation error:
        WH = W*H;
        e = (V-WH)/v; e = sum(e(:)/v);                          %Normalized approximation error
        
        % Break & save conditions:
        t = t+1; E(t,1) = e; %#ok<AGROW>
        if ((e0-e)/e0 <= varmin), break, end
        if flag==1, disp([num2str(t),' : ',num2str(e)]); end
        e0 = e;
        
    end

elseif strcmp(costf,'KLD')                                      %NMF with the generalized Kullback-Leibler divergence
    
    w = sum(W,1); W = W*diag(1./w);                             %Normalization of the components in W
    H = diag(w)*H; H = H*v/sum(H(:));                           %Transfer of the energy of the components of W in H
    WH = W*H; VWH = V./(WH+eps);
    e0 = V.*log(VWH+eps)+WH; e0 = sum(e0(:))/v-1;               %Initial approximation error (iteration 0) (V is subtracted via "-1")
    on = ones(n,1); om = ones(1,m);
    
    while(t < maxiter)
        
        % Update of W:
        W = W.*(VWH*H')./(on*sum(H,2)'+eps);
        w = sum(W,1); W = W*diag(1./w);                         %Normalization of the components in W
        
        % Update of H:
        H = diag(w)*H;                                          %Transfer of the energy in H
        H = H.*(W'*(V./(W*H+eps)))./(sum(W,1)'*om+eps);
        
        % Approximation error:
        WH = W*H; VWH = V./(WH+eps);
        e = V.*log(VWH+eps)+WH; e = sum(e(:))/v-1;              %Normalized approximation error (V is subtracted via "-1")
        
        % Break & save conditions:
        t = t+1; E(t,1) = e; %#ok<AGROW>
        if ((e0-e)/e0 <= varmin), break, end
        if flag==1, disp([num2str(t),' : ',num2str(e)]); end
        e0 = e;

    end

elseif strcmp(costf,'ISD')                                      %NMF with the Itakura-Saito Divergence
    
    w = sum(W,1); W = W*diag(1./w);                             %Normalization of the components in W
    H = diag(w)*H; H = H*v/sum(H(:));                           %Transfer of the energy in H
    WH = W*H+eps; VWH = V./WH+eps;
    e0 = VWH-log(VWH)-1; e0 = sum(e0(:));                       %Initial approximation error (iteration 0)
    
    while(t < maxiter)
        
        % Update of W:
        W = W.*((VWH./WH)*H')./((1./WH)*H');
        w = sum(W,1); W = W*diag(1./w);                         %Normalization of the components in W
        
        % Update of H:
        H = diag(w)*H; %H = H*v/sum(H(:));                      %Transfer of the energy in H
        WH = W*H+eps; H = H.*(W'*(V./(WH.^2)))./(W'*(1./WH));
        
        % Approximation error:
        WH = W*H+eps; VWH = V./WH+eps;
        e = VWH-log(VWH)-1; e = sum(e(:));
        
        % Break & save conditions:
        t = t+1; E(t,1) = e; %#ok<AGROW>
        if ((e0-e)/e0 <= varmin), break, end
        if flag==1, disp([num2str(t),' : ',num2str(e)]); end
        e0 = e;
        
    end

elseif strcmp(costf,'CKL')                                      %KLD + smoothness, sparseness & orthogonality constraints
    
    VWH = V./(W*H+eps);
    e0 = 1.7976e+308;                                           %Initial very large distance
    on = ones(n,1); om = ones(1,m); or = ones(r,1); z = zeros(r,1);
    hs = sum(H,2); hsq = sum(H.^2,2); hhsq = sum((H(:,2:end)-H(:,1:end-1)).^2,2);
    Hsq = hsq*om; Hr = sqrt(H);
    b = prod(1:r)/(2*prod(1:r-2));                              %Binomial coefficient (number of different pairs with r coefficient)
    
    while(t < maxiter)
        
        % Update of W:
        W = W.*(VWH*H')./(on*sum(H,2)'+eps); clear VWH
        w = sum(W,1); W = W*diag(1./w);                         %Normalization of the components in W
        
        % Update of H:
            %"+" gradient:
            Gr = 1;                                             %"+" gradient for the distance parameter
            Gt = 4*m*H./Hsq;                                    %"+" gradient for the smoothness parameter
            Gs = sqrt(m./Hsq);                                  %"+" gradient for the sparseness parameter
            Go = ((or*sum(Hr,1))./(Hr+eps))/(2*b); clear Hr     %"+" gradient for the orthogonality parameter
            Gp = Gr+alpha*Gt+beta*Gs+gamma*Go;                  %"+" total gradient
            
            %"-" gradient:
            Gr = W'*(V./(W*H+eps));                             %"-" gradient for the distance parameter
            Gt = 2*m*(([z,H(:,1:end-1)]+[H(:,2:end),z])./Hsq... %"-" gradient for the smoothness parameter
                +H.*((hhsq./(hsq.^2))*om)); clear Hsq
            Gs = sqrt(m)*H.*((hs./(hsq.^(3/2)))*om);            %"-" gradient for the sparseness parameter
            Go = 1/(2*b);                                       %"-" gradient for the orthogonality parameter
            Gm = Gr+alpha*Gt+beta*Gs+gamma*Go;                  %"-" total gradient
            
            %update:
            H = H.*Gm./(Gp+eps);
            clear Gr Gt Gs Go Gp Gm
            
        % Cost function:
            %distance term (generalized Kullback-Leibler divergence):
            WH = W*H; VWH = V./(WH+eps);
            er = V.*log(VWH+eps)+WH; er = sum(er(:))-v; clear WH      %V is subtracted via "-v"
            
            %smoothness term
            hs = sum(H,2); hsq = sum(H.^2,2); hhsq = sum((H(:,2:end)-H(:,1:end-1)).^2,2);
            Hsq = hsq*om; Hr = sqrt(H);
            et = m*sum(hhsq./hsq);
            
            %sparseness term:
            es = sqrt(m)*sum(hs./sqrt(hsq));
            
            %orthogonality term:
            eo = Hr*Hr'; eo = (sum(eo(:))-sum(H(:)))/(2*b);

            %final approximation error:
            e = (er+alpha*et+beta*es+gamma*eo)/v;
      
        % Break & save conditions:
        t = t+1; E(t,1) = e; E(t,2) = er; E(t,3) = et; E(t,4) = es; E(t,5) = eo; %#ok<AGROW>
        if ((e0-e)/e0 <= varmin), break, end
        if flag==1, disp([num2str(t),' : ',num2str(e)]); end
        e0 = e;
    
    end

end
