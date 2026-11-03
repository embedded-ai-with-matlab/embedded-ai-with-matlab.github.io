function recordings = loadRecordings(rootFolder,sampleRate)
%LOADRECORDINGS Load the Nano 33 BLE Sense recordings onto a uniform time grid.
%   recordings = loadRecordings(rootFolder,sampleRate) reads every CSV file
%   in the class folders under rootFolder. Each file has the columns
%   timestamp_us, ax, ay, az (g), gx, gy, gz (deg/s) and label.
%
%   The logger occasionally misses a sample (a 20 ms step instead of 10 ms),
%   so each recording is linearly interpolated onto a uniform grid at
%   sampleRate (default 100 Hz) using its microsecond timestamps.
%
%   The output table has one row per recording with variables:
%     Signal      - N-by-6 single matrix [ax ay az gx gy gz] at sampleRate
%     Class       - categorical: NORMAL, TAPPING, GENTLE_SHAKING
%     File        - file name
%     Duration    - recording length in seconds
%     MissedSamples - number of samples the logger dropped
%     Gravity     - mean acceleration vector (board orientation), in g

arguments
    rootFolder (1,1) string
    sampleRate (1,1) double = 100
end

classNames = ["NORMAL" "TAPPING" "GENTLE_SHAKING"];
signal = {};
class = strings(0,1);
file = strings(0,1);
duration = zeros(0,1);
missed = zeros(0,1);
gravity = zeros(0,3);

for c = classNames
    files = dir(fullfile(rootFolder,c,"*.csv"));
    for f = files'
        T = readtable(fullfile(f.folder,f.name));
        t = (double(T.timestamp_us) - double(T.timestamp_us(1)))/1e6;
        x = [T.ax T.ay T.az T.gx T.gy T.gz];
        tUniform = (0:1/sampleRate:t(end))';
        signal{end+1,1} = single(interp1(t,x,tUniform,"linear")); %#ok<AGROW>
        class(end+1,1) = c; %#ok<AGROW>
        file(end+1,1) = string(f.name); %#ok<AGROW>
        duration(end+1,1) = t(end); %#ok<AGROW>
        missed(end+1,1) = round(t(end)*sampleRate) + 1 - height(T); %#ok<AGROW>
        gravity(end+1,:) = mean(x(:,1:3)); %#ok<AGROW>
    end
end

recordings = table(signal,categorical(class,classNames),file,duration,missed,gravity, ...
    VariableNames=["Signal" "Class" "File" "Duration" "MissedSamples" "Gravity"]);
end
