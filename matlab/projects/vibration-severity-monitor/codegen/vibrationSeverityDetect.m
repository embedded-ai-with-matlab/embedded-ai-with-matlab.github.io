function [level,probabilities,aeScore,isUnrecognized] = vibrationSeverityDetect(window) %#codegen
%VIBRATIONSEVERITYDETECT Entry-point function compiled to C for the Arduino.
%   [level,probabilities,aeScore,isUnrecognized] = vibrationSeverityDetect(window)
%   takes one 100-by-6 window [ax ay az gx gy gz] from the LSM9DS1 IMU
%   (1 s at 100 Hz, in g and deg/s) and returns:
%
%     level          - vibration level: 0 none, 1 light, 2 strong
%     probabilities  - 3-by-1 classifier probabilities for levels 0, 1, 2
%     aeScore        - autoencoder reconstruction error
%     isUnrecognized - true when the classifier reports level 0 but the
%                      autoencoder does not recognize the window as an
%                      undisturbed board (a minor or unfamiliar vibration)
%
%   Run code generation from the project folder so the MAT files resolve.

persistent netCls netAE
if isempty(netCls)
    netCls = coder.loadDeepLearningNetwork("models/netClsQ.mat");
    netAE = coder.loadDeepLearningNetwork("models/netAEQ.mat");
end
params = coder.load("models/deploymentParams.mat");

% Classifier branch: mean-removed, scaled raw window. The time axis is
% passed as the "T" dimension in CBT layout, which quantized pooling
% layers require for code generation.
x = (single(window) - mean(single(window),1)) ./ params.channelScale;
probabilities = extractdata(predict(netCls,dlarray(reshape(x',6,1,100),"CBT")));
[~,k] = max(probabilities);
level = int32(k - 1);

% Autoencoder branch: 24 motion indicators.
ci = (computeMotionIndicators(window) - params.ciMu) ./ params.ciSigma;
ciHat = extractdata(predict(netAE,dlarray(ci',"CB")));
aeScore = mean((ciHat - ci').^2);

isUnrecognized = level == 0 && aeScore > params.aeThreshold;
end
