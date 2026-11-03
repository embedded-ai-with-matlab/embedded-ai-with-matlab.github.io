% INTRODUCTION: Inspect one IMU recording per class using core MATLAB.
% Start MATLAB in the repository root before running this script.
repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
dataFolder = fullfile(repo,'matlab','projects','vibration-severity-monitor', ...
    'data','raw','Vibration_Dataset');
classes = {'NORMAL','TAPPING','GENTLE_SHAKING'};
results = zeros(numel(classes),3);
figure('Name','Vibration signals by recording condition');
for k = 1:numel(classes)
    filename = fullfile(dataFolder,classes{k},[classes{k} '_001.csv']);
    T = readtable(filename);
    time = (double(T.timestamp_us)-double(T.timestamp_us(1)))*1e-6;
    acceleration = [T.ax T.ay T.az];
    acceleration = acceleration-mean(acceleration,1);
    motion = sqrt(sum(acceleration.^2,2));
    results(k,1) = sqrt(mean(motion.^2));
    results(k,2) = max(motion);
    results(k,3) = median(diff(time))*1e3;
    subplot(3,1,k); plot(time,motion);
    xlabel('Time (s)'); ylabel('Motion (g)'); title(classes{k},'Interpreter','none');
end
summaryTable = table(string(classes'),results(:,1),results(:,2),results(:,3), ...
    'VariableNames',{'Class','CenteredRMS_g','Peak_g','MedianInterval_ms'});
disp(summaryTable);
