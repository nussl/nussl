function [K,q,W,H] = fusion_nmf(W,H,fs,fusion,T)
%
% Fusion function for the "similar" components
%   [K,q,W,H] = fusion_nmf(W,H,fs,fusion,T);
%
% Input:
%   W: Basis matrix
%   H: Weight matrix
%   fs: Sampling frequency (default: fs=44100 Hz)
%   fusion: =1: Fusion from a (unique) measure table
%           =2: Fusion from successive measure tables (default)
%           =3: Fusion from an overlapping table
%   T: Threshold for the fusion (default: T=0.80)
%
% Output:
%   K: Indexes vector of the similar components
%   q: Number of similar components
%   W: Fusionned W (if 4 output arguments)
%   H: Fusionned H (if 4 output arguments)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin<5, T=0.80; end
if nargin<4, fusion=2; end
if nargin<3, fs=44100; end

% Parameters:

r = size(W,2);                                        %Number of components (before fusion)
K = 1:r;
qmax = 5;                                             %Maximum number of fusionned components
qmin = 2;                                             %Minimum number of fusionned components

if r > qmin                                           %Fusion if the number of components > qmin
  q = r;

  if fusion==1                                        %Computation from an unique measure table
    M = spectral_features(W,fs);                      %Spectral features for the components
    M = measure_table(M,2);                           %CSM similarity table between the features
    [v,l,c] = maxi(M);                                %1st max in M (2 most similar features vectors)

    if (q>qmax) || (v>=T)                             %Fusion if q>qmax or if the value is above the threshold
      while (q>qmin)                                  %There will not be less that qmin components remaining after the fusion

        % Indexes of the similar components using the maximum of the table from the highest to the smallest:
        if K(l)~=K(c)                                 %To prevent from fusion within a same cluster
          m = min(K(l),K(c));
          K(find(K==K(l))) = m;                       %All the similar indexes are set to the new fusionned index
          K(find(K==K(c))) = m;
          q = q-1;
        end
        M(l,c) = 0;                                   %The current maximum is set to zero
        [v,l,c] = maxi(M);                            %Next maximum

        if (q<=qmax) && (v<=T), break, end            %Break if q<=qmax & if the value is under the threshold
      end
    end

  elseif fusion==2                                    %Computation of a new measure table after each double fusion

    % 1st features and similarity measure table:
    M = spectral_features(W,fs);                      %Spectral features for the components
    M = measure_table(M,2);                           %CSM similarity table between the features
    O = ones(r);                                      %Matrix to force the indexes of the fusionned components to 0 after fusion
    [v,l,c] = maxi(M);                                %1st max in M (2 most similar features vectors)

    if (q>qmax) || (v>=T)                             %Fusion if q>qmax or if the value is above the threshold
      while (q>qmin)                                  %There will not be less that qmin components remaining after the fusion

        % Indexes of the similar components using the maximum of the table from the highest to the smallest:
        m = min(K(l),K(c));
        K(find(K==K(l))) = m;
        K(find(K==K(c))) = m;
        q = q-1;

        % The components are fusionned to compute a new measure table for the next iteration:
        h1 = H(l,:);
        h2 = H(c,:);
        h = h1+h2;                                    %Fusionned gains (sum!)
        w = (W(:,l)*h1+W(:,c)*h2)*pinv(h);            %Fusionned spectra
        H(l,:) = h; W(:,l) = w;                       %The fusionned components are set with the new index
        W(:,c) = 0; H(c,:) = 0;                       %The W & H of the other index are set to 0 (for the CSM to give null values)
        O(:,c) = 0; O(c,:) = 0;                       %the same for O

        % New features and new similarity measure table for the next iteration:
        M = spectral_features(W);                     %New features
        M = measure_table(M,2);                       %New CSM table
        M = M.*O;                                     %The indexes of the last fusionned components are forced to 0
        [v,l,c] = maxi(M);                            %Next maximum

        if (q<=qmax) && (v<=T), break, end            %If q<=qmax and if the value is under the threshold, break
      end
    end

  elseif fusion==3                                    %Computation from the overlapping measure table (0.025)
    M = overlap_table(H',0);                          %Overlap table from the gains
    [v,l,c] = mini(M);                                %1st min in M (2 most orthogonal gain components)

    if (q>qmax) || (v<=T)                             %Fusion if q>qmax and if the value is under the threshold
      while (q>qmin)                                  %There will not be less that qmin components remaining after the fusion

        if K(l)~=K(c)                                 %To prevent from fusion within a same cluster
          m = min(K(l),K(c));
          K(find(K==K(l))) = m;                       %All the orthogonal indexes are set to the new fusionned index
          K(find(K==K(c))) = m;
          q = q-1;
        end
        M(l,c) = 1;                                   %Setting the current minimum to 1 (max) for the next iteration
        [v,l,c] = mini(M);                            %Next minimum

        if (q<=qmax) && (v>=T), break, end            %If q<=qmax & if the value is above the threshold, break
      end
    end
  end

end

if nargout==4                                         %If 4 output arguments, output fusionned components
  W0 = []; H0 = [];
  for k=1:r                                           %Loop on the initial numbers of components
    f = find(K==k);                                   %Indexes vector of the fusionned components for the number k
    if ~isempty(f)                                    %If f is non empty on K
      w = W(:,f);                                     %Similar spectra for the indexes of f
      h = H(f,:);                                     %Similar gains for the indexes of f
      w = (w*h)*pinv(sum(h,1));                       %Fusion of the spectral components
      h = sum(h,1);                                   %Fusion of the gain components (sum!)
      W0 = [W0,w];                                    %Concatenation in the new fusionned W
      H0 = [H0;h];                                    %Concatenation in the new fusionned H
    end
  end
  W = W0; H = H0;                                     %New fusionned components

end
