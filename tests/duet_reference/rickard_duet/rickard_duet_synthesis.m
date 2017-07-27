%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4. peak centers (determined from histogram)

numsources = 2;
peakdelta = [0 0];
peakalpha=[0 -0.4];

% convert alpha to a
peaka = (peakalpha+sqrt(peakalpha.^2+4))/2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 5. determine masks for separation

bestsofar = Inf*ones(size(tf1));
bestind = zeros(size(tf1));
for i=1:length(peakalpha)
    score = abs(peaka(i)*exp(-sqrt(-1)*fmat*peakdelta(i))...
    .*tf1-tf2).^2/(1+peaka(i)^2);
    mask = (score<bestsofar);
    bestind(mask) = i;
    bestsofar(mask) = score(mask);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 6.& 7. demix with ML alignment and convert to time domain

est = zeros(numsources,length(x1));                             % demixtures
for i=1:numsources
    mask = (bestind==i);
    esti = tfsynthesis([zeros(1,size(tf1,2));
    ((tf1+peaka(i)*exp(sqrt(-1)*fmat*peakdelta(i)).*tf2)...
    ./(1+peaka(i)^2)).*mask],...
    sqrt(2)*awin/1024,timestep,numfreq);
    est(i,:)=esti(1:length(x1))';
    % add back into the demix a little bit of the mixture
    % as that eliminates most of the masking artifacts
    soundsc(est(i,:)+0.05*x1,fs); pause;                        % play demixture
end
