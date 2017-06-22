%   Drag vertical line & integer multiples on a plot (for repet_demo_gui)
%       drag_beat_gui(a,col);
%
%   Input(s):
%       a: axes (default: gca, current axes)
%       col: color of the object to drag (default: 'red')
%
%   See also drag, drag_select, line, patch

%   Author: Zafar RAFII (zafarrafii@u.northwestern.edu)
%   Last update: January 2011

function drag_beat_gui(a,col)

if nargin < 2, col = 'red'; end
if nargin < 1, a = gca; end

f = get(a,'Parent');                                            % Axes' parent = containing figure

set(a, ...
    'HitTest','on', ...                                         % Make sure that axes is clickable
    'ButtonDownFcn',@ClickAxesFcn);                             % When axes is clicked, call ClickAxesFcn

xlim = get(a,'XLim');                                           % Axes' x-limits
xlim(2) = floor(xlim(2)/2);                                     % Half of axes' x-limits
ylim = get(a,'YLim');                                           % Axes' y-limits

c = get(a,'Children');                                          % Axes' children = all graphics within the axes
set(c, ...
    'HitTest','off');                                           % Make children not clickable for axes below to be clickable

l = findobj(a, ...
    'Color',col);                                               % Find vertical lines of color col
set(a, ...
    'UserData',l);                                              % Initialize handles in the axes' userdata

    function ClickAxesFcn(varargin)
        
        coord = get(a,'CurrentPoint');                          % Get coordinates of the mouse click whithin the axes
        if (coord(1,1)<xlim(1) || coord(1,1)>xlim(2) ...        % Return if mouse click not in the horizontal half of the axes box
                || coord(1,2)<ylim(1) || coord(1,2)>ylim(2))
            return
        end
        
        click = get(f, 'SelectionType');                        % Selection type of the mouse click
        if strcmp(click,'normal')                               % Update selection if left click
            delete(l);
        else                                                    % Return if another type of click
            return
        end
        
        coord = coord(1);                                       % x-coordinate
        n = floor(2*xlim(2)/coord);                             % Number of integer multiples of l0
        l = zeros(n,1);
        
        l(n) = line(coord*ones(1,2),ylim, ...                   % Main line
            'Color',col, ...
            'LineStyle','-', ...
            'LineWidth',1, ...
            'HitTest','on', ...                                 % Make sure that the main line is clickable
            'ButtonDownFcn',@ClickLineFcn);                     % When main line clicked, call ClickLineFcn
        
        hold on
        for i = 2:n
            l(n-i+1) = line(coord*ones(1,2)*i,ylim, ...         % Integer multiples (sorted in decreasing order because lines are "queued")
                'Color',col, ...
                'LineStyle',':', ...
                'LineWidth',1, ...
                'HitTest','off');                               % Make sure that the integer multiples are not clickable
        end
        hold off
                
        set(f, ...
            'WindowButtonMotionFcn',@DragLineFcn, ...           % When mouse moves over the figure, call DragLinFcn
            'WindowButtonUpFcn',@ReleaseFcn);                   % When mouse button is released from the figure, call ReleaseFcn
        
        
        function ClickLineFcn(varargin)
            
            click = get(f, 'SelectionType');
            if strcmp(click,'normal')
                set(f, ...
                    'WindowButtonMotionFcn',@DragLineFcn);      % When mouse moves over the figure, call DragLineFcn
            end
            
        end
        
        
        function DragLineFcn(varargin)
            
            coord = get(a, 'CurrentPoint');
            if (coord(1,1)<xlim(1) || coord(1,1)>xlim(2) ...
                    || coord(1,2)<ylim(1) || coord(1,2)>ylim(2))
                return
            end
            
            coord = coord(1);
            n = floor(2*xlim(2)/coord);
            delete(l)                                           % Delete previous lines
            l = zeros(n,1);
            
            l(n) = line(coord*ones(1,2),ylim, ...               % Reset main line
                'Color',col, ...
                'LineStyle','-', ...
                'LineWidth',1, ...
                'HitTest','on', ...
                'ButtonDownFcn',@ClickLineFcn);
            
            hold on
            for j = 2:n
                l(n-j+1) = line(coord*ones(1,2)*j,ylim, ...     % Reset integer multiples
                    'Color',col, ...
                    'LineStyle',':', ...
                    'LineWidth',1, ...
                    'HitTest','off');
            end
            hold off
            
        end
        
        
        function ReleaseFcn(varargin)
            
            coord = get(a,'CurrentPoint');
            if (coord(1,1)<xlim(1) || coord(1,1)>xlim(2) ...
                    || coord(1,2)<ylim(1) || coord(1,2)>ylim(2))
                return
            end
            
            set(f, ...
                'WindowButtonMotionFcn','');                    % When mouse moves, do nothing
            
            set(a, ...
                'UserData',l);                                  % Save handles in the axes' userdata
            
        end
        
    end

end
