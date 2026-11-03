function [windows,recordingIndex] = makeWindows(signals,windowLength,hop)
%MAKEWINDOWS Cut recordings into fixed-length windows.
%   [windows,recordingIndex] = makeWindows(signals,windowLength,hop) cuts
%   each N-by-6 recording in the cell array signals into windowLength-by-6
%   windows that start every hop samples. recordingIndex gives, for each
%   window, the index of the recording it came from, so that data can be
%   split by recording instead of by window.

windows = {};
recordingIndex = zeros(0,1);
for r = 1:numel(signals)
    x = signals{r};
    for s = 1:hop:size(x,1) - windowLength + 1
        windows{end+1,1} = x(s:s+windowLength-1,:); %#ok<AGROW>
        recordingIndex(end+1,1) = r; %#ok<AGROW>
    end
end
end
