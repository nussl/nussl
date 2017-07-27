%   Beat spectrum (= mean over the autocorrelations of the rows)
%       B = beat_spectrum(X);
%
%   Input(s):
%       X: spectrogram [n x time]
%
%   Output(s):
%       B: beat spectrum [1 x lag]
%
%   See also repet, beat_spectrogram

%   Author: Zafar RAFII (zafarrafii@u.northwestern.edu)
%   Last update: February 2011

function B = beat_spectrum(X)

opt = 'unbiased';
[n,m] = size(X);

B = zeros(n,m);
for i = 1:n                                                     % Loop on the rows of the spectrogram
    Bi = xcorr(X(i,:),opt);                                     % Autocorrelation of the rows
    B(i,:) = Bi(:,m:2*m-1);                                     % Autocorrelation from lag 0 to lag m-1
end
B = mean(B,1);                                                  % Mean along the columns

% % Foote's version 1:
% 
% M = similarity_matrix(X,2,1,0);                                 % Similarity matrix using the Cosine similarity measure
% 
% m = size(M,1);
% B = zeros(1,m);
% for j = 1:m
%     B(j) = mean(diag(M,-j+1));                                  % Mean over the diagonals
% end
% 
% % Foote's version 2:
% 
% M = similarity_matrix(X,2,1,0);                                 % Similarity matrix using the Cosine similarity measure
% M1 = padarray(M,[m,m],0,'post');
% M2 = padarray(M,[m,m],0,'pre');
% 
% m = size(M,1);
% B = zeros(1,m);
% for j = 1:m
%     M = M1((1:m)+j-1,(1:m)+j-1).*M2((1:m)+m-j+1,(1:m)+m-j+1);
%     B(j) = sum(M(:))/(((m-j+1)*(m-j+1+1))/2);                   % Autocorrelation of the diagonals (with zero-padding)
% end
