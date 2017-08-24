function [frq_path, autoCorr_path, time, amp] = praat_pd(in,fs,display,time_step,min_frq,...
    max_frq,VoiceThresh,SilThresh,OctaveCost,VoicedUnvoicedCost,OctaveJumpCost,...
    Hypotheses)

% [frq_path, autoCorr_path, time, amp] = praat_pd(in,fs,display,time_step,min_frq,...
%      max_frq,VoiceThresh,SilThresh,OctaveCost,VoicedUnvoicedCost,OctaveJumpCost,...
%      Hypotheses)
%  
%      Thus function takes a vector of linearly encoded PCM values and
%      returns a pitch track of them, storing the returned frequencies in
%      frq_path.
%  
%      Example usage:  
%         load laughter;
%         display = 1;
%         [frq_path, autoCorr_path, time, amp] = praat_pd(y,Fs,display);
%  
%  
%   Output values
%   ----------------
%     frq_path      - a vector of values where frq_path(i) is the best guess
%                   of the  frequency of the fundamental pitch at time(i).
%  
%     autoCorr_path - a vector where autoCorr_path(i) is strength 
%                   of autocorrelation in the window at time step i. This is 
%                   basically a measure of harmonicity and ranges from 0 to 1.
%  
%     time          - a vector of values where time(i) is the time at which
%                   frq_path(i)is determined (measured in seconds).
%  
%     amp           - a vector of amplitudes, where amp(i) is the amplitude of
%                   the signal at time(i).
%  
%  
%   Input arguments 
%   ---------------------------------------------------------
%   NOTE: Other than "in" all input arguments are optional. That said, if you
%   want to specify the ith argument, you have to specify all arguments j <
%   i. For example, if you specify "time_step," then you have to specify "in",
%   "fs", and "display" before specifying "time_step".
%  
%   in                  - a vector of values assumed to be linearly encoded, 
%                         PCM audio
%   fs                  - a scalar that specifies sample frequency of the 
%                         input audio (default 44,100 Hz)
%   display             - a scalar, if set to 1, a display of the output is 
%                         created (default is 0)
%   time_step           - a scalar, the time between window centers of the 
%                         ffts used to find the pitch track(default is 0.01)
%   min_frq             - a scalar, the lowest frequency hypothesis allowed 
%                         (also affects window size of FFTs) (default is 30 Hertz)
%   max_frq             - a scalar, the highest frequency hypothesis allowed  
%                         (default is 1000 Hertz)
%   VoiceThresh         - a scalar that varies what counts as "voiced" (harmonic)  
%                         (default is 0.5, range is 0 to 1)
%   SilThresh           - a scalar that sets the maximum for what counts as  
%                         silence (default is 0.02  range is 0 to 1)
%   OctaveCost          - (default is 0.02)
%   VoicedUnvoicedCost  - (default is 0.2)
%   OctaveJumpCost      - (default is 0.2)
%   Hypotheses          - How many different pitch hypotheses will be considered
%                         (default is 4)
%   
% 
% Original code by Mark Bartsch
% Based on praat by Paul Boersma.
% Additional tweaks, comments, etc. by Bryan Pardo.

warning off;

% this bit of code simply sets default values if arguments are missing
% from the input

if nargin < 2 | isempty(fs)
    fs = 44100;
end
if nargin < 3 | isempty(display)
    display = 0;
end
if nargin < 4 | isempty(time_step)
    time_step = 0.01;
end
if nargin < 5 | isempty(min_frq)
    min_frq = 30;
end
if nargin < 6 | isempty(max_frq)
    max_frq = 5000;
end
if nargin < 7 | isempty(VoiceThresh)
%    VoiceThresh = 0.4;
    VoiceThresh = 0.5;
end
if nargin < 8 | isempty(SilThresh)
%    SilThresh = 0.05;
    SilThresh = 0.02;
end
if nargin < 9 | isempty(OctaveCost)
    OctaveCost = 0.02;
end
if nargin < 10 | isempty(VoicedUnvoicedCost)
    VoicedUnvoicedCost = 0.2;
end
if nargin < 11 | isempty(OctaveJumpCost)
    OctaveJumpCost = 0.9;
end
if nargin < 12 | isempty(Hypotheses)
    Hypotheses = 4;
end

% STEP 1: skip Boersma's preprocessing step.
% ------
in = in(:);
% z = fft(in);
% ind = round(length(z)*.475):ceil(length(z)/2);
% z(ind) = z(ind).*linspace(1,0,length(ind))';
% z(length(z)-ind) = z(length(z)-ind).*linspace(1,0,length(ind))';
% z = [z(1:floor(length(z)/2)); 0; z(ceil(length(z)/2):length(z))];
% in = ifft(z);
% 
%  The above is a "preprocessing step" in Boersma. Try it and see if it
%  makes any difference. Just uncomment these lines.


% calculate the distance between fft window centers in samples
samp_step = round(time_step*fs);
% figure out the needed window size (in samples, given the lowest
%  frequency we want to be able to capture
window_size = round(3*fs/min_frq);
% now figure out the nearest power of two larger than the window size,
% since ffts need windows whose size is a power of 2
fft_size = 2^(ceil(log2(window_size*1.5)));
fft_pad = fft_size - window_size;
% figure out how much adjacent windows will overlap
overlap = window_size-samp_step;

% STEP 2: this is the largest (loudest) sample in the audio
% -------
global_peak = max(abs(in));

% Boersma suggests a gaussian window, but my Mark Bartsch (who wrote the original
% version of this file) claims hanning is just fine. I'll go along with that, since 
% Mark is usually right. If you'd like to try the guassian window, just
% uncomment it and comment the line that uses the hanning window.
%window = gausswin(window_size,2.47)
window = hanning(window_size);
% figure out what the autocorrelation function of the window is
window_autoCorr = fft([window; zeros(fft_pad,1)]);
window_autoCorr = window_autoCorr.*conj(window_autoCorr);
window_autoCorr = real(ifft(window_autoCorr));
window_autoCorr = window_autoCorr./window_autoCorr(1);

% calculate where the fft window centers will be, in terms of samples 
t = (1:samp_step:length(in)-window_size)';

%don't worry about display stuff
if display
    sg = zeros(fft_size/2+1,length(t));
    autoCorr_mat = zeros(floor(window_size/2),length(t));
end

% this just lets you know the computation is in progress.
%disp('computing pitch-track');

% set up some variables to fill them with data in a few lines
amp = zeros(length(t),1);
lags = zeros(length(t),Hypotheses);
autoCorr_vals = zeros(length(t),Hypotheses-1);
scores = zeros(length(t),Hypotheses);

% this is the main loop in the function, and for every time in the vector
% "t" it figures out the pitch
counter = 1;
for time = t'
    
    % STEP 3.1 : Grab a  section of the signal
    % grab a section of the audio equal to the size of one window, starting
    % at the ith time in t
    frame = in(time:time+window_size-1);
    % STEP 3.2: Subtract the local average value
    amp(counter) = sqrt(mean((frame-mean(frame)).^2));
    
    % STEPS 3.4 and 3.5 and 3.6 and 3.7: multiply by the window function, then 
    % append 1/2
    % a window length of 0s (because we need autocorrelation values up to
    % 1/2 a window length for interpolation)... When padding, make sure the
    % window is also long enough to make the number of samples a power of
    % 2. This lets a FFT be omputed.
    spec = fft([(frame - mean(frame)).*window; zeros(fft_pad,1)]);
    % STEP 3.8 square the samples in the frequency domain
    % multiply it by its complex conjugate
    spec = spec.*conj(spec);
    % now take the real part of the INVERSE fft, giving the autocorrelation
    autoCorr = real(ifft(spec));
    
    % Divide auocorrelation of the windowed signal by the
    % autocorrelation of the window
    % Normalize so things range from 0 to 1
    autoCorr = autoCorr/autoCorr(1)./window_autoCorr;
    
    % everything in this "if" statement is to help with data display
    if display
        sg(:,counter) = spec(1:fft_size/2+1);
        autoCorr_mat(:,counter) = autoCorr(1:floor(window_size/2));
    end
    
    % look for local maxima in the autocorrelation matrix
    [lags(counter,1:3),autoCorr_vals(counter,1:3)] = ...
        peak_search(autoCorr,Hypotheses-1,fs/max_frq,fs/min_frq,window_size/2,OctaveCost);
    
    loc_peak = max(frame);
    scores(counter,4) = VoiceThresh + max([0, 2-(loc_peak/global_peak)/(SilThresh/(1+VoiceThresh))]);
    scores(counter,1:3) = autoCorr_vals(counter,1:3) - OctaveCost*log2(min_frq*lags(counter,1:3)/fs);
    
    %wb = waitbar_txt(time/t(end),wb);
    counter = counter + 1;
end
fprintf(2,'\n');
scores(isinf(scores)) = -inf;

frqs = fs./lags;
frqs(isinf(frqs)) = 0;


% STEP 4. Implement Viterbi path finding
% This stuff tries to find a path from frame to frame that minimizes the
% transition costs, taking into account the octave jump costs. Essentially,
% for every frame, there are four top candidates for the pitch. This path
% searching triest o find the most "coherent" path from frame to frame to
% come up with an estimate for the pitch that doesn't jump around too much
% as it goes through the sound.

last_cost = -scores(1,:);
this_cost = zeros(1,size(frqs,2));
paths = zeros(size(frqs));
for time = 2:size(frqs,1)
    for this = 1:size(frqs,2)
        temp_cost = zeros(1,size(frqs,2));
        for that = 1:size(frqs,2)
            if xor(frqs(time-1,that) == 0,frqs(time,this) == 0)
                temp_cost(that) = last_cost(that) + VoicedUnvoicedCost - scores(time,this);
            elseif and(frqs(time-1,that),frqs(time,this))
                cst = OctaveJumpCost*abs(log2(frqs(time-1,that)/frqs(time,this)));
                temp_cost(that) = last_cost(that) + cst - scores(time,this);
            else
                temp_cost(that) = last_cost(that) - scores(time,this);
            end
        end
        [this_cost(this),paths(time,this)] = min(temp_cost);
    end
    last_cost = this_cost;
    this_cost = zeros(size(this_cost));
end

% now we are ready to create the output for this function
% first, make a couple of empty arrays to hold the values
frq_path = zeros(size(frqs,1),1);
autoCorr_path = zeros(size(frq_path));
[mn,last_choice] = min(last_cost);

% now fill them up with the best choices for frequencies and
% autocorrelation, given the optimal path found with the Viterbi algorithm.
for i=length(frq_path):-1:1
    if last_choice == Hypotheses
        autoCorr_path(i) = 0;
    else
        autoCorr_path(i) = autoCorr_vals(i,last_choice);
    end
    frq_path(i) = frqs(i,last_choice);
    last_choice = paths(i,last_choice);
end    

h2n = 10*log10(autoCorr_path./(1-autoCorr_path));
lag_path = fs./(frq_path);

% The stuff in this if-end statement is display stuff, you don't need to 
% worry about it, unless you'd like to learn more about making images in
% MATLAB. Note the use of the log scaling for the spectrogram
if display
    figure(1);
    imagesc([1 size(sg,2)],[0 fs/2],10*log10(sg)); axis xy
    title('Spectrogram'); colorbar
    hold on
    pth_plot = frq_path;
    pth_plot(pth_plot == 0) = -inf;
    plot(pth_plot,'kx-');
    hold off;
    xlabel('Frame number');
    ylabel('Frequency (Hz)');
    
    figure(2)
    imagesc(autoCorr_mat);
    axis xy;
    hold on
    plot(lag_path,'kx-');
    hold off;
    title('Autocorrelation'); colorbar
    xlabel('Frame number');
    ylabel('Lag (in samples)');
    
end

t = t + floor(window_size/2);
time = t * (1/fs);
warning on


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For each frame, this function finds the number of peaks defined in
% "num_peaks". IE it finds the top local maxima in the function

function [peaks,ampl] = peak_search(x,num_peaks,ind_min,ind_max,max_lag,OctCost)

%  peak_search(autoCorr,Hypotheses-1,fs/max_frq,fs/min_frq,window_size/2,Oc
%  taveCost);
% arguments
% ----------
% x         - the function, encoded as a vector. As used in this matlab
%             file, x = the windowed auto correlation "autoCorr"
% num_peaks - the number of peaks to return. As used in this function, this
%             is based on the number of allowed pitch track hypotheses 
% ind_min   - this is set to the sample frequency divided by the maximum
%             allowed frequency hypothesis. This gives the index of the
%             minimal allowed frequency
% ind_min   - this is set to the sample frequency divided by the minimum
%             allowed frequency hypothesis. This gives the index of the
%             minimal allowed frequency
% max_lag   - look at the correlation values up to 1/2 a window away.
% OctCost   - the penalty paid for being in a higher octave.
%
% returns
% -------
% peaks     - a vector of peak autocorrelation freqencies
% ampl      - a vector of amplitudes where the ith amplitude corresponds to
%               the ith peak
%

peak_sep = ind_min;
ind_min = round(ind_min);

peaks = ones(1,num_peaks);
ampl = ones(1,num_peaks)*1e-10;

for fst_neg=1:ind_max
    if x(fst_neg) < 0
        break
    end
end
ind_min = max([fst_neg ind_min]);

% find the index numbers of the local maxima
inds = find(diff(sign(diff(x((ind_min-1):(ind_max+1))))) == -2) - 1 + ind_min;
% for each local maximum, assign a cost based on octave
cost = -(x(inds) - OctCost*log2(inds/ind_max));
% sort the peaks by cost
[srt,ind] = sort(cost);
ind = inds(ind);
ind(x(ind) < 0) = [];

for ct = 1:num_peaks
    if isempty(ind)
        return
    end   
    xval = [ind(1)-2; ind(1)-1; ind(1)];
    A = [xval.^2 xval [1; 1; 1]];
    b = x(xval+1);
    y = A\b;
    peaks(ct) = -y(2)/2/y(1);
    ampl(ct) = y(1)*(peaks(ct))^2 + y(2)*peaks(ct) + y(3);
    
    ind(abs(ind(1) - ind) < peak_sep) = [];
end

