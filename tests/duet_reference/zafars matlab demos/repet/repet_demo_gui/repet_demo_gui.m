function varargout = repet_demo_gui(varargin)
% repet_demo_gui M-file for repet_demo_gui.fig
%      repet_demo_gui, by itself, creates a new repet_demo_gui or raises the existing
%      singleton*.
%
%      H = repet_demo_gui returns the handle to a new repet_demo_gui or the handle to
%      the existing singleton*.
%
%      repet_demo_gui('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in repet_demo_gui.M with the given input arguments.
%
%      repet_demo_gui('Property','Value',...) creates a new repet_demo_gui or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before repet_demo_gui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to repet_demo_gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help repet_demo_gui

% Last Modified by GUIDE v2.5 17-Feb-2011 18:19:03

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @repet_demo_gui_OpeningFcn, ...
                   'gui_OutputFcn',  @repet_demo_gui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before repet_demo_gui is made visible.
function repet_demo_gui_OpeningFcn(hObject, eventdata, handles, varargin) %#ok<*INUSL>
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to repet_demo_gui (see VARARGIN)

% Choose default command line output for repet_demo_gui
handles.output = hObject;

p = triu(ones(5,6),1);                                          % Upper triangular '1' matrix/lower triangular '0' matrix
p = [p*233/255;zeros(1,6);flipud(p*211/255)];                   % [11x6] '0' play icon (on a gradient gray background)
play_icon = zeros(11,11);                                       % [11x10] stretched play icon
play_icon(:,1:2:end) = p(:,1:end);
play_icon(:,2:2:end) = p(:,2:end);
play_icon = repmat(play_icon,[1,1,3]);                          % Play icon image

set(handles.pushbutton_play_mixture,'CData',play_icon);
set(handles.pushbutton_play_foreground,'CData',play_icon);
set(handles.pushbutton_play_background,'CData',play_icon);

stop_icon = zeros(11,11,3);                                     % Stop icon image
set(handles.pushbutton_stop_mixture,'CData',stop_icon);
set(handles.pushbutton_stop_foreground,'CData',stop_icon);
set(handles.pushbutton_stop_background,'CData',stop_icon);

colormap(hot);                                                  % Varies from black - through red, orange, and yellow - to white

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes repet_demo_gui wait for user response (see UIRESUME)
% uiwait(handles.figure_repet);


% --- Outputs from this function are returned to the command line.
function varargout = repet_demo_gui_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton_load_mixture.
function pushbutton_load_mixture_Callback(hObject, eventdata, handles) %#ok<*DEFNU>
% hObject    handle to pushbutton_load_mixture (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Load mixture file:

[filename,filepath] = uigetfile( ...
    {'*.wav;*.mp3', 'WAV & MP3 files (*.wav, *.mp3)'; ...
    '*.wav', 'WAV files only (*.wav)'; ...
    '*.mp3', 'MP3 files only (*.mp3)'; ...
    '*.*',  'All Files (*.*)'}, ...
    'Select a WAV or MP3 file...');
if isequal(filename,0)                                          % If 'Cancel', return
    return
end

[~,filename,fileext] = fileparts(filename);
if ~(strcmp(fileext,'.wav') || strcmp(fileext,'.mp3'))          % If extension unknown, return
    return
end

watchon;                                                        % Set pointer to the watch
drawnow;

% Disable/reset toolbar:

set(handles.uitoggletool_select, ...
    'Enable','off', ...
    'State','on');
set(handles.uitoggletool_zoom, ...
    'Enable','off', ...
    'State','off');
zoom off
set(handles.uitoggletool_pan, ...
    'Enable','off', ...
    'State','off');
pan off

% Disable mixture controls:

set(handles.pushbutton_load_mixture, ...
    'Enable','off');
set(handles.pushbutton_play_mixture, ...
    'Enable','off');
set(handles.pushbutton_stop_mixture, ...
    'Enable','off');
set(handles.pushbutton_beat_mixture, ...
    'Enable','off');

% Reset mixture wave, spectrogram & beat spectrum axes:

axes_mixture_wave = handles.axes_mixture_wave;
cla(axes_mixture_wave,'reset');                                 % Reset axes
set(axes_mixture_wave, ...                                      % Because axis ticks are still visible after reset
    'Box','on', ...
    'HitTest','off', ...
    'XTick',[], ...
    'YTick',[]);
axes_mixture_spec = handles.axes_mixture_spec;
cla(axes_mixture_spec,'reset');
set(axes_mixture_spec, ...
    'Box','on', ...
    'HitTest','off', ...
    'XTick',[], ...
    'YTick',[]);
axes_mixture_beat = handles.axes_mixture_beat;
cla(axes_mixture_beat,'reset');
set(axes_mixture_beat, ...
    'Box','on', ...
    'HitTest','off', ...
    'XTick',[], ...
    'YTick',[]);

% Disable/reset mixture unmix: 

set(handles.slider_period_mixture, ...
    'Enable','off');
set(handles.edit_period_mixture, ...
    'Enable','off', ...
    'String','');
set(handles.text_period_mixture, ...
    'Enable','off', ...
    'Min',0, ... 
    'Max',1, ... 
    'SliderStep',[0.01,0.1], ...
    'Value',0);
set(handles.slider_tolerance_mixture, ...
    'Enable','off', ...
    'Min',0, ... 
    'Max',1, ... 
    'SliderStep',[0.01,0.1], ...
    'Value',0);
set(handles.edit_tolerance_mixture, ...
    'Enable','off', ...
    'String','');
set(handles.text_tolerance_mixture, ...
    'Enable','off');
set(handles.pushbutton_unmix_mixture, ...
    'Enable','off');

% Reset output:

reset_output(handles);

% Read mixture wave:

file = fullfile(filepath,[filename,fileext]);                   % File with path & extension
if strcmp(fileext,'.wav')
    [x,fs,nbits] = wavread(file);
elseif strcmp(fileext,'.mp3')
    [x,fs,nbits] = mp3read(file);                               % Use mp3 toolbox
end

if length(filename) > 30                                        % Shorten if name has more than 30 characters
    filename = [filename(1:30),'~'];
end

L = length(x);                                                  % Wave length (in samples)
if L > 15*fs                                                    % Truncate to random 20 sec and mono, if more than 20 sec
    i = round(rand*(L-15*fs));
    x = mean(x((1:15*fs)+i,:),2);
end

a = audioplayer(x,fs,nbits);                                    % Initialize audioplayer

% Enable toolbar:

set(handles.uitoggletool_select, ...
    'Enable','on');
set(handles.uitoggletool_zoom, ...
    'Enable','on');
set(handles.uitoggletool_pan, ...
    'Enable','on');

% Enable mixture controls:

set(handles.pushbutton_load_mixture, ...
    'Enable','on');
set(handles.pushbutton_play_mixture, ...
    'Enable','on');
set(handles.pushbutton_stop_mixture, ...
    'Enable','on');
set(handles.pushbutton_beat_mixture, ...
    'Enable','on');

% Plot mixture wave:

L = length(x);                                                  % Wave length (in samples)
time_wave = (1:L)/fs;                                           % Wave time vector (in sec) of length L
axes(axes_mixture_wave); %#ok<*MAXES>
plot(time_wave,x);
set(axes_mixture_wave, ...
    'HitTest','on', ...
    'XLim',[1,L]/fs, ...
    'XGrid','on', ...
    'YLim',[-1,1]);
title(filename,'Interpreter','none')                            % No interpreter to avoid underscore = subscript
xlabel('time (sec)')
ylabel('amplitude')
zoom reset

drag_select(axes_mixture_wave);                                 % Make mixture wave axes draggable (boundaries are saved in axes' userdata)

% Save handles (global variables):

handles.filename = filename;
handles.x = x;
handles.fs = fs;
handles.nbits = nbits;
handles.L = L;
handles.a = a;

watchoff;                                                       % Set pointer back to the arrow
drawnow;

guidata(hObject,handles);                                       % Update handles structure


function reset_output(handles)                                  % To be used after load, beat, or unmix, right after watchon

% Disable foreground controls:

set(handles.pushbutton_save_foreground, ...
    'Enable','off');
set(handles.pushbutton_play_foreground, ...
    'Enable','off');
set(handles.pushbutton_stop_foreground, ...
    'Enable','off');

% Reset foreground axes:

axes_foreground_wave = handles.axes_foreground_wave;
cla(axes_foreground_wave,'reset');
set(axes_foreground_wave, ...
    'Box','on', ...
    'HitTest','off', ...
    'XTick',[], ...
    'YTick',[]);
axes_foreground_spec = handles.axes_foreground_spec;
cla(axes_foreground_spec,'reset');
set(axes_foreground_spec, ...
    'Box','on', ...
    'HitTest','off', ...
    'XTick',[], ...
    'YTick',[]);
axes_foreground_beat = handles.axes_foreground_beat;
cla(axes_foreground_beat,'reset');
set(axes_foreground_beat, ...
    'Box','on', ...
    'HitTest','off', ...
    'XTick',[], ...
    'YTick',[]);

% Disable background controls:

set(handles.pushbutton_save_background, ...
    'Enable','off');
set(handles.pushbutton_play_background, ...
    'Enable','off');
set(handles.pushbutton_stop_background, ...
    'Enable','off');

% Reset background axes:

axes_background_wave = handles.axes_background_wave;
cla(axes_background_wave,'reset');
set(axes_background_wave, ...
    'Box','on', ...
    'HitTest','off', ...
    'XTick',[], ...
    'YTick',[]);
axes_background_spec = handles.axes_background_spec;
cla(axes_background_spec,'reset');
set(axes_background_spec, ...
    'Box','on', ...
    'HitTest','off', ...
    'XTick',[], ...
    'YTick',[]);
axes_background_beat = handles.axes_background_beat;
cla(axes_background_beat,'reset');
set(axes_background_beat, ...
    'Box','on', ...
    'HitTest','off', ...
    'XTick',[], ...
    'YTick',[]);


% --- Executes on button press in pushbutton_play_mixture.
function pushbutton_play_mixture_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_play_mixture (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

axes_mixture_wave = handles.axes_mixture_wave;
fs = handles.fs;
L = handles.L;
a = handles.a;

lims = get(axes_mixture_wave, 'UserData');                      % Get current selection boundaries
if lims(1) == lims(2)                                           % If just one line, the second boundary is the right x-limit for the playing
    lims(2) = L/fs;                                             % Not saved in handles!
end

set(handles.pushbutton_load_mixture, ...
    'Enable','off');                                            % Load button off while playing

col = [84,84,84]/255;                                           % Gray color
if isplaying(a)
    stop(a);
    l = findobj(axes_mixture_wave, ...                          % Need to findobj marker line because l is not saved in handles
        'Type','line', ...
        'Color',col);
    delete(l);                                                  % Delete previous marker line before re-playing
end

play(a,round(lims*fs));                                         % Play selection
axes(axes_mixture_wave);
l = line([1,1]*lims(1),[-1,1], ...                              % Initialize marker line
    'LineStyle','-', ...
    'Color',col, ...
    'LineWidth',1, ...
    'HitTest','off');                                           % Marker line not clickable for the patch to be clickable

while isplaying(a)
    i = get(a, 'CurrentSample');                                % Get current played sample   
    set(l, ...
        'XData',[1,1]*i/fs);                                    % Update marker line
    drawnow;
end

l = findobj(axes_mixture_wave, ...                              % Need again to findobj marker line because l is not saved in handles
    'Color',col);
delete(l);                                                      % Delete marker line when done playing

set(handles.pushbutton_load_mixture, ...
    'Enable','on');

guidata(hObject,handles);


% --- Executes on button press in pushbutton_stop_mixture.
function pushbutton_stop_mixture_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_stop_mixture (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

a = handles.a;

if isplaying(a)
    stop(a);
    set(handles.pushbutton_load_mixture, ...
        'Enable','on');
end

guidata(hObject,handles);


% --- Executes on button press in pushbutton_beat_mixture.
function pushbutton_beat_mixture_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_beat_mixture (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

watchon;
drawnow;

% Disable toolbar:

set(handles.uitoggletool_select, ...
    'Enable','off');
set(handles.uitoggletool_zoom, ...
    'Enable','off');
set(handles.uitoggletool_pan, ...
    'Enable','off');

% Disable mixture controls:

set(handles.pushbutton_load_mixture, ...
    'Enable','off');
set(handles.pushbutton_play_mixture, ...
    'Enable','off');
set(handles.pushbutton_stop_mixture, ...
    'Enable','off');
set(handles.pushbutton_beat_mixture, ...
    'Enable','off');

% Disable mixture wave, and reset spectrogram & beat spectrum axes:

axes_mixture_wave = handles.axes_mixture_wave;
set(axes_mixture_wave, ...
    'HitTest','off');
linkaxes(axes_mixture_wave,'off');
axes_mixture_spec = handles.axes_mixture_spec;
cla(axes_mixture_spec,'reset');
set(axes_mixture_spec, ...
    'Box','on', ...
    'HitTest','off', ...
    'XTick',[], ...
    'YTick',[]);
axes_mixture_beat = handles.axes_mixture_beat;
cla(axes_mixture_beat,'reset');
set(axes_mixture_beat, ...
    'Box','on', ...
    'HitTest','off', ...
    'XTick',[], ...
    'YTick',[]);

% Disable/reset mixture unmix: 

set(handles.slider_period_mixture, ...
    'Enable','off');
set(handles.edit_period_mixture, ...
    'Enable','off', ...
    'String','');
set(handles.text_period_mixture, ...
    'Enable','off', ...
    'Min',0, ... 
    'Max',1, ... 
    'SliderStep',[0.01,0.1], ...
    'Value',0);
set(handles.slider_tolerance_mixture, ...
    'Enable','off', ...
    'Min',0, ... 
    'Max',1, ... 
    'SliderStep',[0.01,0.1], ...
    'Value',0);
set(handles.edit_tolerance_mixture, ...
    'Enable','off', ...
    'String','');
set(handles.text_tolerance_mixture, ...
    'Enable','off');
set(handles.pushbutton_unmix_mixture, ...
    'Enable','off');

% Reset output:

reset_output(handles);

% Load handles:

x = handles.x;
fs = handles.fs;
L = handles.L;

% Mixture wave selection:

lims = get(axes_mixture_wave, 'UserData');                      % Boundaries (in sec)
if lims(1) == lims(2)                                           % If just one line, the boundaries are the x-limits for beat
    lims(1) = 1;                                                % Not saved in handles!
    lims(2) = L;
else
    lims = round(lims*fs);                                      % Boundaries (in samples)
end
lim = lims(1):lims(2);                                          % Wave selection time vector (in samples)

N = 2^nextpow2(fs*0.04);                                        % Analysis window length (next power of 2) (= 40 msec for music signals)
overlap = N/2;
step = N-overlap;
m = ceil((length(lim)-overlap)/step);                           % Spectrogram length
if m < 5                                                        % floor(m-1/2) should at least 1
    return
end

time_wave = lim/fs;                                             % Wave selection time vector (in sec)
x = x(lim,:);                                                   % Wave selection vector (in sec)

% Mixture STFT & beat spectrum:

X = stft(x,hamming(N),overlap);                                 % Short-Time Fourier Transform
n = N/2+1;
V = abs(X(1:n,:,:));                                            % Spectrogram including DC component (for beat spectrum)

b = beat_spectrum(mean(V.^2,3));                                % Beat spectrum of the mean power spectrogram
b = b/b(1);                                                     % Normalization

% Estimate optimal period & tolerance:

per = repet_period(b);                                          % Optimal period (in samples)
tol = 0;                                                        % Optimal tolerance

% Mixture spectrogram & beat spectrum plot parameters:

S = 20*log10(abs(mean(X(2:N/2+1,:,:),3)));                      % Spectrogram without DC component (in dB) averaged over the channels (for imagesc)
time_spec = (lims(1):(lims(2)-lims(1))/(m-1):lims(2))/fs;       % Spectrogram time vector (in sec)
freq = (1:n-1)*(fs/N)*1e-3;                                     % Spectrogram frequency vector (in kHz) (n included the DC component)
clims = [mini(S),maxi(S)];                                      % Color value limits (for the foreground and background spectrograms later)

time_beat = (1:(lims(2)-lims(1))/(m-1):(lims(2)-lims(1)))/fs;   % Beat spectrum time vector (in sec)

% Enable toolbar:

set(handles.uitoggletool_select, ...
    'Enable','on');
set(handles.uitoggletool_zoom, ...
    'Enable','on');
set(handles.uitoggletool_pan, ...
    'Enable','on');

% Enable mixture controls:

set(handles.pushbutton_load_mixture, ...
    'Enable','on');
set(handles.pushbutton_play_mixture, ...
    'Enable','on');
set(handles.pushbutton_stop_mixture, ...
    'Enable','on');
set(handles.pushbutton_beat_mixture, ...
    'Enable','on');

% Enable mixture wave:

set(axes_mixture_wave, ...
    'HitTest','on');
axes(axes_mixture_wave);
xlabel('')                                                      % Clear wave axes xlabel (to save space)

% Plot mixture spectrogram: 

axes(axes_mixture_spec);
imagesc(time_spec,freq,S,clims);
set(axes_mixture_spec, ...
    'HitTest','on', ...
    'XLim',[1,L]/fs, ...
    'XGrid','on', ...
    'YLim',[freq(1),freq(n-1)])
title('Spectrogram (dB)');
xlabel('time (sec)');
ylabel('frequency (kHz)');
zoom reset
linkaxes([axes_mixture_wave,axes_mixture_spec],'x');            % Synchronize x-axis limits of spectrogram axes with wave axes (for zoom and pan)

% Plot mixture beat spectrum: 

axes(axes_mixture_beat);
plot(time_beat,b(2:m),'black');                                 % Beat spectrum without lag 0 (time_beat does not have lag 0)
set(axes_mixture_beat, ...
    'HitTest','on', ...
    'XLim',[time_beat(1),time_beat(m-1)], ...
    'YLim',[0,1]);
title('Beat Spectrum')
xlabel('time lag (sec)')
ylabel('correlation')

% Enable/set mixture unmix:

m2 = floor((m-1)/2);                                            % Half of the time_beat length
set(handles.slider_period_mixture, ...
    'Enable','on', ...
    'Min',time_beat(1), ... 
    'Max',time_beat(m2), ... 
    'SliderStep',[1,10]/(m2-1), ...                             % Minor & major slider step (1 sample & 10 samples, translated in sec)
    'Value',time_beat(per));                                    % Period (in sec)
set(handles.edit_period_mixture, ...
    'Enable','on', ...
    'String',roundto(time_beat(per),3), ...                     % Edit's display rounded to 3 decimals
    'UserData',per);                                            % Save period (in samples) in edit's userdata because of function handles and callbacks
set(handles.text_period_mixture, ...
    'Enable','on');

tol_min = -1;
tol_max = 1;
set(handles.slider_tolerance_mixture, ...
    'Enable','on', ...
    'Min',tol_min, ... 
    'Max',tol_max, ... 
    'SliderStep',[0.1,1]/(tol_max-tol_min), ...
    'Value',tol);
set(handles.edit_tolerance_mixture, ...
    'Enable','on', ...
    'String',roundto(tol,1));                                   % Edit's display rounded to 1 decimal
set(handles.text_tolerance_mixture, ...
    'Enable','on');

set(handles.pushbutton_unmix_mixture, ...
    'Enable','on');

% Save handles:

handles.X = X;
handles.V = V;
handles.clims = clims;
handles.b = b;
handles.time_wave = time_wave;
handles.time_spec = time_spec;
handles.time_beat = time_beat;

draggable_beat(hObject, eventdata, handles);                    % Create draggable mixture beat spectrum

watchoff;
drawnow;

guidata(hObject,handles);


function draggable_beat(hObject, eventdata, handles)

axes_mixture_beat = handles.axes_mixture_beat;
time_beat = handles.time_beat;

set(axes_mixture_beat, ...
    'HitTest','on', ...                                         % Make sure that mixture beat axes is clickable
    'ButtonDownFcn',{@ClickMixtureBeatAxesFcn,handles});        % When mixture beat axe is clicked, call ClickMixtureBeatAxesFcn

axes_mixture_beat_chilren = get(axes_mixture_beat,'Children');  % Axes' children = all graphics within the axes
set(axes_mixture_beat_chilren, ...
    'HitTest','off');                                           % Make children not clickable for axes below to be clickable

per = get(handles.edit_period_mixture, 'UserData');             % Period in samples from edit's userdata
m = length(time_beat);                                          % Length of the beat spectrum (without lag 0) = length of the spectrogram - 1 
r = floor(m/per);                                               % Number of integer multiples of the period in the beat spectrum (without lag 0)

axes(axes_mixture_beat);
hold on
beat = line(repmat(time_beat(per:per:r*per),[2,1]),repmat([0;1],[1,r]), ... % Initialize period marker & integer multiples on the beat spectrum (sorted in decreasing order)
    'LineStyle',':', ...
    'Color','red', ...
    'LineWidth',1, ...
    'HitTest','off');
set(beat(1), ...
    'LineStyle','-', ...                                        % Period marker in solid line
    'HitTest','on', ...                                         % Make sure that the period marker is clickable
    'ButtonDownFcn',{@ClickMixtureBeatLineFcn,handles});        % When period marker is clicked, call ClickMixtureBeatLineFcn
hold off

set(axes_mixture_beat, ...                                      % Save beat in axes' userdata because handles cannot be "shared" between "sub callbacks"
    'UserData',beat);

guidata(hObject,handles);

function ClickMixtureBeatAxesFcn(src, eventdata, handles)       % src and eventdata are the two default inputs for a function handle

figure_repet = handles.figure_repet;
axes_mixture_beat = handles.axes_mixture_beat;
time_beat = handles.time_beat;

coord = get(axes_mixture_beat, 'CurrentPoint');                 % Get coordinates of the mouse click whithin the axes
m = length(time_beat);
if (coord(1,1)<time_beat(1) || coord(1,1)>time_beat(floor(m/2)) ... % Return if mouse click is not in the horizontal half of the axes box
        || coord(1,2)<0 || coord(1,2)>1)
    return
end

click = get(figure_repet, 'SelectionType');                     % Selection type of the mouse click
if strcmp(click,'normal')                                       % Update selection if left click
    beat = get(axes_mixture_beat, 'UserData');
    delete(beat);
else                                                            % Return if another type of click
    return
end

coord = coord(1);                                               % x-coordinate (in sec)
[~,per] = min(abs(time_beat-coord));                            % Period (in samples) (closest index)
r = floor(m/per);

axes(axes_mixture_beat);
hold on
beat = line(repmat(time_beat(per:per:r*per),[2,1]),repmat([0;1],[1,r]), ...% New period marker & integer multiples (cannot simply update because of integer multiples)
    'LineStyle',':', ...
    'Color','red', ...
    'LineWidth',1, ...
    'HitTest','off');
set(beat(1), ...
    'LineStyle','-', ...
    'HitTest','on', ...
    'ButtonDownFcn',{@ClickMixtureBeatLineFcn,handles});
hold off

set(figure_repet, ...
    'WindowButtonMotionFcn',{@DragMixtureBeatLineFcn,handles}, ...  % When mouse moves over the figure, call DragMixtureBeatLineFcn
    'WindowButtonUpFcn',{@ReleaseMixtureBeatLineFcn,handles});  % When mouse button is released from the figure, call ReleaseMixtureBeatLineFcn

set(handles.slider_period_mixture, ...
    'Value',time_beat(per));                                    % Update per (in sec) in slider (not rounded)
set(handles.edit_period_mixture, ...
    'String',num2str(roundto(time_beat(per),3)), ...            % Update per (in sec) in edit (rounded to 3 decimals)
    'UserData',per);                                            % Save per (in samples) in edit's userdata because handles cannot be "shared" between "sub callbacks"
set(axes_mixture_beat, ...
    'UserData',beat);

function ClickMixtureBeatLineFcn(src, eventdata, handles)

figure_repet = handles.figure_repet;

click = get(figure_repet, 'SelectionType');
if strcmp(click,'normal')
    set(figure_repet, ...
        'WindowButtonMotionFcn',{@DragMixtureBeatLineFcn,handles}); % When mouse moves over the figure, call DragMixtureBeatLineFcn
end

function DragMixtureBeatLineFcn(src, eventdata, handles)

axes_mixture_beat = handles.axes_mixture_beat;
time_beat = handles.time_beat;

coord = get(axes_mixture_beat, 'CurrentPoint');
m = length(time_beat);
if (coord(1,1)<time_beat(1) || coord(1,1)>time_beat(floor(m/2)) ...
        || coord(1,2)<0 || coord(1,2)>1)
    return
end

coord = coord(1);
[~,per] = min(abs(time_beat-coord));
r = floor(m/per);

beat = get(axes_mixture_beat, 'UserData');
delete(beat);

axes(axes_mixture_beat);
hold on
beat = line(repmat(time_beat(per:per:r*per),[2,1]),repmat([0;1],[1,r]), ...
    'LineStyle',':', ...
    'Color','red', ...
    'LineWidth',1, ...
    'HitTest','off');
set(beat(1), ...
    'LineStyle','-', ...
    'HitTest','on', ...
    'ButtonDownFcn',{@ClickMixtureBeatLineFcn,handles});
hold off

set(handles.slider_period_mixture, ...
    'Value',time_beat(per));
set(handles.edit_period_mixture, ...
    'String',num2str(roundto(time_beat(per),3)), ...
    'UserData',per);
set(axes_mixture_beat, ...
    'UserData',beat);

function ReleaseMixtureBeatLineFcn(src, eventdata, handles)

figure_repet = handles.figure_repet;

set(figure_repet, ...
    'WindowButtonMotionFcn','');                                % When mouse moves, do nothing


% --- Executes on slider movement.
function slider_period_mixture_Callback(hObject, eventdata, handles)
% hObject    handle to slider_period_mixture (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

axes_mixture_beat = handles.axes_mixture_beat;
time_beat = handles.time_beat;

per = get(handles.slider_period_mixture, 'Value');              % Period (in sec)
[~,per] = min(abs(time_beat-per));                              % Period (in samples) (closest index)

m = length(time_beat);
r = floor(m/per);

beat = get(axes_mixture_beat, 'UserData');
delete(beat)

axes(axes_mixture_beat);
hold on
beat = line(repmat(time_beat(per:per:r*per),[2,1]),repmat([0;1],[1,r]), ... % Update period marker & integer multiples
    'LineStyle',':', ...
    'Color','red', ...
    'LineWidth',1, ...
    'HitTest','off');
set(beat(1), ...
    'LineStyle','-', ...
    'HitTest','on', ...
    'ButtonDownFcn',{@ClickMixtureBeatLineFcn,handles});
hold off

set(handles.slider_period_mixture, ...
    'Value',time_beat(per));
set(handles.edit_period_mixture, ...
    'String',num2str(roundto(time_beat(per),3)), ...
    'UserData', per);
set(axes_mixture_beat, ...
    'UserData',beat);

guidata(hObject,handles);


% --- Executes during object creation, after setting all properties.
function slider_period_mixture_CreateFcn(hObject, eventdata, handles) %#ok<*INUSD>
% hObject    handle to slider_period_mixture (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


function edit_period_mixture_Callback(hObject, eventdata, handles)
% hObject    handle to edit_period_mixture (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_period_mixture as text
%        str2double(get(hObject,'String')) returns contents of edit_period_mixture
%        as a double

axes_mixture_beat = handles.axes_mixture_beat;
time_beat = handles.time_beat;

per = get(handles.edit_period_mixture, 'String');               % Period (string)
per = str2double(per);                                          % Period (in sec) (= NaN if not string of a number)

if isnan(per)                                                   % If per is not a number, reset edit from slider and return
    per = get(handles.slider_period_mixture, 'Value');
    set(handles.edit_period_mixture, ...
        'String',roundto(per,3));
    return
end

m = length(time_beat);
m2 = floor(m/2);                                                % Half of the time_beat length
[~,per] = min(abs(time_beat(1:m2)-per));                        % Period (in samples) (closest index) (values out of range would be rescaled)
r = floor(m/per);

beat = get(axes_mixture_beat, 'UserData');
delete(beat)

axes(axes_mixture_beat);
hold on
beat = line(repmat(time_beat(per:per:r*per),[2,1]),repmat([0;1],[1,r]), ... % Update period marker & integer multiples
    'LineStyle',':', ...
    'Color','red', ...
    'LineWidth',1, ...
    'HitTest','off');
set(beat(1), ...
    'LineStyle','-', ...
    'HitTest','on', ...
    'ButtonDownFcn',{@ClickMixtureBeatLineFcn,handles});
hold off

set(handles.slider_period_mixture, ...
    'Value',time_beat(per));
set(handles.edit_period_mixture, ...
    'String',num2str(roundto(time_beat(per),3)), ...
    'UserData', per);
set(axes_mixture_beat, ...
    'UserData',beat);

guidata(hObject,handles);


% --- Executes during object creation, after setting all properties.
function edit_period_mixture_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_period_mixture (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on slider movement.
function slider_tolerance_mixture_Callback(hObject, eventdata, handles)
% hObject    handle to slider_tolerance_mixture (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

tol = get(handles.slider_tolerance_mixture, 'Value');

set(handles.slider_tolerance_mixture, ...
    'Value',roundto(tol,1));
set(handles.edit_tolerance_mixture, ...
    'String',num2str(roundto(tol,1)));

guidata(hObject,handles);


% --- Executes during object creation, after setting all properties.
function slider_tolerance_mixture_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider_tolerance_mixture (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


function edit_tolerance_mixture_Callback(hObject, eventdata, handles)
% hObject    handle to edit_tolerance_mixture (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_tolerance_mixture as text
%        str2double(get(hObject,'String')) returns contents of edit_tolerance_mixture as a double

tol_min = get(handles.slider_tolerance_mixture, 'Min');
tol_max = get(handles.slider_tolerance_mixture, 'Max');

tol = get(handles.edit_tolerance_mixture, 'String');
tol = str2double(tol);

if isnan(tol)                                                   % If tol is not a number, reset edit from slider and return
    tol = get(handles.slider_tolerance_mixture, 'Value');
    set(handles.edit_tolerance_mixture, ...
        'String',roundto(tol,1));
    return
end

if tol < tol_min                                                % If tol out of range, rescale
    tol = tol_min;
elseif tol > tol_max
    tol = tol_max;
end

set(handles.slider_tolerance_mixture, ...
    'Value',roundto(tol,1));
set(handles.edit_tolerance_mixture, ...
    'String',num2str(roundto(tol,1)));

guidata(hObject,handles);


% --- Executes during object creation, after setting all properties.
function edit_tolerance_mixture_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_tolerance_mixture (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton_unmix_mixture.
function pushbutton_unmix_mixture_Callback(hObject, ~, handles)
% hObject    handle to pushbutton_unmix_mixture (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

watchon;
drawnow;

% Disable/reset toolbar:

set(handles.uitoggletool_select, ...
    'Enable','off', ...
    'State','on');
set(handles.uitoggletool_zoom, ...
    'Enable','off', ...
    'State','off');
zoom off
set(handles.uitoggletool_pan, ...
    'Enable','off', ...
    'State','off');
pan off

% Disable mixture controls:

set(handles.pushbutton_load_mixture, ...
    'Enable','off');
set(handles.pushbutton_play_mixture, ...
    'Enable','off');
set(handles.pushbutton_stop_mixture, ...
    'Enable','off');
set(handles.pushbutton_beat_mixture, ...
    'Enable','off');

% Disable mixture wave, spectrogram & beat spectrum axes:

axes_mixture_wave = handles.axes_mixture_wave;
linkaxes(axes_mixture_wave,'off');
set(axes_mixture_wave, ...
    'HitTest','off');
axes_mixture_spec = handles.axes_mixture_spec;
linkaxes(axes_mixture_spec,'off');
set(axes_mixture_spec, ...
    'HitTest','off');
axes_mixture_beat = handles.axes_mixture_beat;
set(axes_mixture_beat, ...
    'HitTest','off');

% Disable mixture unmix: 

set(handles.slider_period_mixture, ...
    'Enable','off');
set(handles.edit_period_mixture, ...
    'Enable','off');
set(handles.text_period_mixture, ...
    'Enable','off');
set(handles.slider_tolerance_mixture, ...
    'Enable','off');
set(handles.edit_tolerance_mixture, ...
    'Enable','off');
set(handles.text_tolerance_mixture, ...
    'Enable','off');
set(handles.pushbutton_unmix_mixture, ...
    'Enable','off');

% Reset output:

reset_output(handles);

% Load handles:

fs = handles.fs;
nbits = handles.nbits;
L = handles.L;
X = handles.X;
V = handles.V;
clims = handles.clims;
time_wave = handles.time_wave;
time_spec = handles.time_spec;
time_beat = handles.time_beat;

% figure_repet & Masking:

per = get(handles.edit_period_mixture, 'UserData');
tol = get(handles.edit_tolerance_mixture, 'String');
tol = str2double(tol);

M = repet_core(V,per,tol);
M(M==0) = 0.1;
M(M==1) = 0.9;
[N,m,~] = size(X);
n = N/2+1;
M = [M;M(n-1:-1:2,:,:)];                                        % Symmetrize binary mask

X1 = X.*(1-M);                                                  % STFT of the non-repeating foreground
X2 = X-X1;                                                      % STFT of the repeating background

% Parameterization for foreground/background wave, spectrogram, & beat spectrum:

l = length(time_wave);
x1 = istft(X1,hamming(N),N/2);
x1 = x1(1:l,:);
x2 = istft(X2,hamming(N),N/2);
x2 = x2(1:l,:);

S1 = 20*log10(abs(mean(X1(2:N/2+1,:,:),3)));
S2 = 20*log10(abs(mean(X2(2:N/2+1,:,:),3)));
freq = (1:n-1)*(fs/N)*1e-3;

V1 = abs(X1(1:N/2+1,:,:));
b1 = beat_spectrum(mean(V1.^2,3));
b1 = b1/b1(1);
V2 = abs(X2(1:N/2+1,:,:));
b2 = beat_spectrum(mean(V2.^2,3));
b2 = b2/b2(1);

% Enable toolbar:

set(handles.uitoggletool_select, ...
    'Enable','on');
set(handles.uitoggletool_zoom, ...
    'Enable','on');
set(handles.uitoggletool_pan, ...
    'Enable','on');

% Enable mixture controls:

set(handles.pushbutton_load_mixture, ...
    'Enable','on');
set(handles.pushbutton_play_mixture, ...
    'Enable','on');
set(handles.pushbutton_stop_mixture, ...
    'Enable','on');
set(handles.pushbutton_beat_mixture, ...
    'Enable','on');

% Disable mixture wave, spectrogram & beat spectrum axes:

set(axes_mixture_wave, ...
    'HitTest','on');
set(axes_mixture_spec, ...
    'HitTest','on');
set(axes_mixture_beat, ...
    'HitTest','on');

% Disable mixture unmix: 

set(handles.slider_period_mixture, ...
    'Enable','on');
set(handles.edit_period_mixture, ...
    'Enable','on');
set(handles.text_period_mixture, ...
    'Enable','on');
set(handles.slider_tolerance_mixture, ...
    'Enable','on');
set(handles.edit_tolerance_mixture, ...
    'Enable','on');
set(handles.text_tolerance_mixture, ...
    'Enable','on');
set(handles.pushbutton_unmix_mixture, ...
    'Enable','on');

% Enable foreground controls:

set(handles.pushbutton_save_foreground, ...
    'Enable','on');
set(handles.pushbutton_play_foreground, ...
    'Enable','on');
set(handles.pushbutton_stop_foreground, ...
    'Enable','on');

% Plot foreground wave:

axes_foreground_wave = handles.axes_foreground_wave;
axes_foreground_spec = handles.axes_foreground_spec;
axes_foreground_beat = handles.axes_foreground_beat;

axes(axes_foreground_wave);
plot(time_wave,x1);
set(axes_foreground_wave, ...
    'HitTest','on', ...
    'XLim',[1/fs,L/fs], ...
    'XGrid','on', ...
    'YLim',[-1,1]);
title('Non-repeating Foreground')
ylabel('amplitude')

drag_select(axes_foreground_wave);
set(axes_foreground_wave, ...
    'UserData',[time_wave(1),time_wave(l)]);                    % Overwrite axes' boundaries with selection limits

% Plot foreground spectrogram:

axes(axes_foreground_spec);
imagesc(time_spec,freq,S1,clims);
set(axes_foreground_spec, ...
    'HitTest','on', ...
    'XLim',[1/fs,L/fs], ...
    'XGrid','on', ...
    'YLim',[freq(1),freq(n-1)])
title('Spectrogram (dB)')
xlabel('time (sec)');
ylabel('frequency (kHz)');

% Plot foreground beat spectrum:

axes(axes_foreground_beat);
plot(time_beat,b1(2:m),'black');
set(axes_foreground_beat, ...
    'HitTest','on', ...
    'XLim',[time_beat(1),time_beat(m-1)], ...
    'YLim',[0,1]);
title('Beat Spectrum')
xlabel('time lag (sec)')
ylabel('correlation')

% Enable background controls:

set(handles.pushbutton_save_background, ...
    'Enable','on');
set(handles.pushbutton_play_background, ...
    'Enable','on');
set(handles.pushbutton_stop_background, ...
    'Enable','on');

% Plot background wave:

axes_background_wave = handles.axes_background_wave;
axes_background_spec = handles.axes_background_spec;
axes_background_beat = handles.axes_background_beat;

axes(axes_background_wave);
plot(time_wave,x2);
set(axes_background_wave, ...
    'HitTest','on', ...
    'XLim',[1/fs,L/fs], ...
    'XGrid','on', ...
    'YLim',[-1,1]);
title('Repeating Background')

drag_select(axes_background_wave);
set(axes_background_wave, ...
    'UserData',[time_wave(1),time_wave(l)]);                    % Overwrite axes' boundaries with selection limits

% Plot background spectrogram:

axes(axes_background_spec);
imagesc(time_spec,freq,S2,clims);
set(axes_background_spec, ...
    'HitTest','on', ...
    'XLim',[1/fs,L/fs], ...
    'XGrid','on', ...
    'YLim',[freq(1),freq(n-1)])
title('Spectrogram (dB)')
xlabel('time (sec)');

% Plot background beat spectrum:

axes(axes_background_beat);
plot(time_beat,b2(2:m),'black');
set(axes_background_beat, ...
    'HitTest','on', ...
    'XLim',[time_beat(1),time_beat(m-1)], ...
    'YLim',[0,1]);
title('Beat Spectrum')
xlabel('time lag (sec)')

axes_mixture_wave = handles.axes_mixture_wave;
axes_mixture_spec = handles.axes_mixture_spec;

linkaxes([axes_mixture_wave, axes_mixture_spec, ...
    axes_foreground_wave, axes_foreground_spec, ...
    axes_background_wave, axes_background_spec],'x');

% Foreground & background audioplayers:

a1 = audioplayer(x1,fs,nbits);
a2 = audioplayer(x2,fs,nbits);

% Save in handles (global variables):

handles.x1 = x1;
handles.x2 = x2;
handles.a1 = a1;
handles.a2 = a2;

watchoff;
drawnow;

guidata(hObject,handles);


% --- Executes on button press in pushbutton_save_foreground.
function pushbutton_save_foreground_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_save_foreground (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

filename = handles.filename;
fs = handles.fs;
nbits = handles.nbits;
x1 = handles.x1;

[filename,pathname] ...
    = uiputfile({'*.wav';'*.mp3'}, ...
    'Save as a wav or mp3 file', [filename,'_foreground']);
if ~isequal(filename,0) || ~isequal(pathname,0)
    file = [pathname,filename];
    [~,~,fileext] = fileparts(file);
    if ~strcmp(fileext,'.wav') && ~strcmp(fileext,'.mp3')
        return
    end
else
    return
end

if strcmp(fileext,'.wav')
    wavwrite(x1,fs,nbits,file);
elseif strcmp(fileext,'.mp3')
    mp3write(x1,fs,nbits,file);
end

guidata(hObject, handles);


% --- Executes on button press in pushbutton_play_foreground.
function pushbutton_play_foreground_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_play_foreground (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

axes_foreground_wave = handles.axes_foreground_wave;
fs = handles.fs;
time_wave = handles.time_wave;
a1 = handles.a1;

lims = get(axes_foreground_wave, 'UserData');
if (lims(1) < time_wave(1) && lims(2) <= time_wave(1)) ...
        || (lims(1) >= time_wave(end) && lims(2) >= time_wave(end)) % If the foreground selection does not at least intersects the audio, return
    return
end
if lims(1) < time_wave(1)                                       % Force selection start to be at least the audio start
    lims(1) = time_wave(1);
end
if lims(2) > time_wave(end)                                     % Force selection end to be at most the audio end
    lims(2) = time_wave(end);
end
if lims(1) == lims(2)
    lims(2) = time_wave(end);
end
delta = lims(1)-1/fs;                                           % Time difference (in sec) between the selection start and the audio start

set(handles.pushbutton_load_mixture, ...
    'Enable','off');
set(handles.pushbutton_unmix_mixture, ...
    'Enable','off');

col = [84,84,84]/255;                                           % Gray color
if isplaying(a1)
    stop(a1);
    l = findobj(axes_foreground_wave, ...
        'Type','line', ...
        'Color',col);
    delete(l);
end

play(a1,round((lims-delta)*fs));
axes(axes_foreground_wave);
l = line([1,1]*lims(1),[-1,1], ...
    'LineStyle','-', ...
    'Color',col, ...
    'LineWidth',1, ...
    'HitTest','off');

while isplaying(a1)
    i = get(a1, 'CurrentSample');
    set(l, ...
        'XData',[1,1]*i/fs+delta);
    drawnow
end

l = findobj(axes_foreground_wave, ...
    'Color',col);
delete(l);

set(handles.pushbutton_load_mixture, ...
    'Enable','on');
set(handles.pushbutton_unmix_mixture, ...
    'Enable','on');

guidata(hObject, handles);


% --- Executes on button press in pushbutton_stop_foreground.
function pushbutton_stop_foreground_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_stop_foreground (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

a1 = handles.a1;

if isplaying(a1)
    stop(a1);
    set(handles.pushbutton_load_mixture, ...
        'Enable','on');
    set(handles.pushbutton_unmix_mixture, ...
        'Enable','on');
end

guidata(hObject, handles);


% --- Executes on button press in pushbutton_save_background.
function pushbutton_save_background_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_save_background (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

filename = handles.filename;
fs = handles.fs;
nbits = handles.nbits;
x2 = handles.x2;

[filename,pathname] ...
    = uiputfile({'*.wav';'*.mp3'}, ...
    'Save as a wav or mp3 file', [filename,'_background']);
if ~isequal(filename,0) || ~isequal(pathname,0)
    file = [pathname,filename];
    [~,~,fileext] = fileparts(file);
    if ~strcmp(fileext,'.wav') && ~strcmp(fileext,'.mp3')
        return
    end
else
    return
end

if strcmp(fileext,'.wav')
    wavwrite(x2,fs,nbits,file);
elseif strcmp(fileext,'.mp3')
    mp3write(x2,fs,nbits,file);
end

guidata(hObject, handles);


% --- Executes on button press in pushbutton_play_background.
function pushbutton_play_background_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_play_background (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

axes_background_wave = handles.axes_background_wave;
fs = handles.fs;
time_wave = handles.time_wave;
a2 = handles.a2;

lims = get(axes_background_wave, 'UserData');
if (lims(1) < time_wave(1) && lims(2) <= time_wave(1)) ...
        || (lims(1) >= time_wave(end) && lims(2) >= time_wave(end))
    return
end
if lims(1) < time_wave(1)
    lims(1) = time_wave(1);
end
if lims(2) > time_wave(end)
    lims(2) = time_wave(end);
end
if lims(1) == lims(2)
    lims(2) = time_wave(end);
end
delta = lims(1)-1/fs;

set(handles.pushbutton_load_mixture, ...
    'Enable','off');
set(handles.pushbutton_unmix_mixture, ...
    'Enable','off');

col = [84,84,84]/255;
if isplaying(a2)
    stop(a2);
    l = findobj(axes_background_wave, ...
        'Type','line', ...
        'Color',col);
    delete(l);
end

play(a2,round((lims-delta)*fs));
axes(axes_background_wave);
l = line([1,1]*lims(1),[-1,1], ...
    'LineStyle','-', ...
    'Color',col, ...
    'LineWidth',1, ...
    'HitTest','off');

while isplaying(a2)
    i = get(a2, 'CurrentSample');
    set(l, ...
        'XData',[1,1]*i/fs+delta);
    drawnow
end

l = findobj(axes_background_wave, ...
    'Color',col);
delete(l);

set(handles.pushbutton_load_mixture, ...
    'Enable','on');
set(handles.pushbutton_unmix_mixture, ...
    'Enable','on');

guidata(hObject, handles);


% --- Executes on button press in pushbutton_stop_background.
function pushbutton_stop_background_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_stop_background (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

a2 = handles.a2;

if isplaying(a2)
    stop(a2);
    set(handles.pushbutton_load_mixture, ...
        'Enable','on');
    set(handles.pushbutton_unmix_mixture, ...
        'Enable','on');
end

guidata(hObject, handles);


% --------------------------------------------------------------------
function uitoggletool_select_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to uitoggletool_select (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

state = get(handles.uitoggletool_select, 'State');
if strcmp(state,'off')                                          % If select gets unpushed while already pushed (1 becomes 0)
    set(handles.uitoggletool_select, ...
        'State','on');                                          % Push it back (zoom and pan already unpushed)
    
elseif strcmp(state,'on')                                       % If select gets pushed while unpushed (0 becomes 1)
    set(handles.uitoggletool_zoom, ...
        'State','off');                                         % Unpush zoom
    set(handles.uitoggletool_pan, ...
        'State','off');                                         % Unpush pan
    set(handles.axes_mixture_wave, ...
        'HitTest','on');
    set(handles.axes_mixture_beat, ...
        'HitTest','on');
    set(handles.axes_foreground_wave, ...
        'HitTest','on');
    set(handles.axes_foreground_beat, ...
        'HitTest','on');
    set(handles.axes_background_wave, ...
        'HitTest','on');
    set(handles.axes_background_beat, ...
        'HitTest','on');
end

zoom off
pan off

guidata(hObject, handles);


% --------------------------------------------------------------------
function uitoggletool_zoom_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to uitoggletool_zoom (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

state = get(handles.uitoggletool_zoom, 'State');
if strcmp(state,'off')                                          % If zoom gets unpushed while already pushed (1 becomes 0)
    set(handles.uitoggletool_zoom, ...
        'State','on');                                          % Push it back (select and pan already unpushed)
    
elseif strcmp(state,'on')                                       % If zoom gets pushed while unpushed (0 becomes 1)
    set(handles.uitoggletool_select, ...
        'State','off');                                         % Unpush select
    set(handles.uitoggletool_pan, ...
        'State','off');                                         % Unpush pan
    
    set(handles.axes_mixture_wave, ...
        'HitTest','off');
    set(handles.axes_mixture_beat, ...
        'HitTest','off');
    set(handles.axes_foreground_wave, ...
        'HitTest','off');
    set(handles.axes_foreground_beat, ...
        'HitTest','off');
    set(handles.axes_background_wave, ...
        'HitTest','off');
    set(handles.axes_background_beat, ...
        'HitTest','off');
end

zoom xon
pan off

guidata(hObject, handles);


% --------------------------------------------------------------------
function uitoggletool_pan_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to uitoggletool_pan (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

state = get(handles.uitoggletool_pan, 'State');
if strcmp(state,'off')                                          % If pan gets unpushed while already pushed (1 becomes 0)
    set(handles.uitoggletool_pan, ...
        'State','on');                                          % Push it back (select and zoom already unpushed)
    
elseif strcmp(state,'on')                                       % If pan gets pushed while unpushed (0 becomes 1)
    set(handles.uitoggletool_select, ...
        'State','off');                                         % Unpush select
    set(handles.uitoggletool_zoom, ...
        'State','off');                                         % Unpush zoom
    
    set(handles.axes_mixture_wave, ...
        'HitTest','off');
    set(handles.axes_mixture_beat, ...
        'HitTest','off');
    set(handles.axes_foreground_wave, ...
        'HitTest','off');
    set(handles.axes_foreground_beat, ...
        'HitTest','off');
    set(handles.axes_background_wave, ...
        'HitTest','off');
    set(handles.axes_background_beat, ...
        'HitTest','off');
end

zoom off
pan xon

guidata(hObject, handles);
