% LAB 1: Train a decision-tree baseline on prepared vibration windows.
% Start MATLAB in the repository root and run Part1_PrepareData.m first.
repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
projectRoot = fullfile(repo,'matlab','projects','vibration-severity-monitor');
load(fullfile(projectRoot,'data','preparedData.mat'), ...
    'windowsTrain','windowsTest','labelsTrain','labelsTest');
% Inputs are one 100x6 (or equivalently time x channels) window per cell.
% We compute centered acceleration RMS on the first three columns.
trainRMS = featureRMS(windowsTrain);
testRMS = featureRMS(windowsTest);
% Log-transform and fit a shallow classifier (Statistics and ML Toolbox).
trainX = log10(max(trainRMS,eps));
testX  = log10(max(testRMS,eps));
model = fitctree(trainX,labelsTrain,'MaxNumSplits',2);
prediction = predict(model,testX);
accuracy = mean(prediction==labelsTest);
fprintf('Recording-held-out test accuracy: %.2f%%\n',accuracy*100);
figure('Name','Lab 1 baseline confusion');
confusionchart(labelsTest,prediction);
% Local helper functions are supported in modern MATLAB script files.
function values = featureRMS(windows)
    n = numel(windows);
    values = zeros(n,1);
    for i = 1:n
        x = double(windows{i}(:,1:3));
        x = x-mean(x,1);
        values(i) = sqrt(mean(sum(x.^2,2)));
    end
end
