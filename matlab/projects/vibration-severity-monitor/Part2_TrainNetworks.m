%[text] # Part 2: Train the Vibration Severity Networks
%[text] This script first measures how far a simple threshold gets, then trains the two networks of the hybrid detector:
%[text] - a **1-D CNN classifier** that reads the raw 6-axis window and reports the vibration level (0, 1, or 2), and
%[text] - an **autoencoder**, trained only on `NORMAL` windows, that flags any vibration that does not look like an undisturbed board. \
%[text] Run **Part1\_PrepareData** first.
%%
%[text] ## Load the Prepared Data
projectFolder = fileparts(mfilename("fullpath"));
cd(projectFolder)
addpath(fullfile(projectFolder,"helpers"),fullfile(projectFolder,"codegen"))
load(fullfile(projectFolder,"data","preparedData.mat"))
rng(0)
classNames = categories(labelsTrain);
forDisplay = @(c) renamecats(c,"GENTLE_SHAKING","GENTLE SHAKING");   % avoid TeX subscript in charts
%%
%[text] ## Start with a Baseline: An RMS Threshold
%[text] Before training a network, check what a simple rule achieves. Fit a decision tree with two splits on the log RMS acceleration, which amounts to choosing two thresholds.
rmsOf = @(w) cellfun(@(x) rms(reshape(x(:,1:3) - mean(x(:,1:3)),[],1)),w);
baseline = fitctree(log10(rmsOf(windowsTrain)),labelsTrain,MaxNumSplits=2);
predictedBaseline = predict(baseline,log10(rmsOf(windowsTest)));
accuracyBaseline = 100*mean(predictedBaseline == labelsTest)
confusionchart(forDisplay(labelsTest),forDisplay(predictedBaseline),Title="RMS threshold baseline (test set)")
%[text] Vibration energy alone already separates most windows, because the three classes differ a lot in level. Its mistakes are at the boundary between light and strong vibration: a vigorous tap carries as much energy as a gentle shake. Telling them apart needs the *shape* of the signal, an isolated impulse versus continuous motion, which is what the CNN learns.
%%
%[text] ## Prepare the Classifier Input
%[text] The classifier reads each window as a sequence of 100 time steps with 6 channels. Remove the mean of each channel (gravity and gyroscope bias) and divide by the per-channel scale computed in Part 1.
prepare = @(w) cellfun(@(x) (x - mean(x,1))./channelScale,w,UniformOutput=false);
XTrain = prepare(windowsTrain);
XVal = prepare(windowsVal);
XTest = prepare(windowsTest);
%%
%[text] ## Define and Train the 1-D CNN
%[text] The architecture matches the first tutorial: three convolution blocks, global average pooling, and two fully connected layers. Global pooling makes the prediction independent of *where* in the window a tap occurs.
layersCls = [
    sequenceInputLayer(6,MinLength=100)
    convolution1dLayer(5,16,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(2,Stride=2)
    convolution1dLayer(5,32,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(2,Stride=2)
    convolution1dLayer(3,32,Padding="same")
    batchNormalizationLayer
    reluLayer
    globalAveragePooling1dLayer
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(3)
    softmaxLayer];
optionsCls = trainingOptions("adam", ...
    MaxEpochs=40, ...
    MiniBatchSize=64, ...
    InitialLearnRate=2e-3, ...
    InputDataFormats="TCB", ...
    ValidationData={XVal,labelsVal}, ...
    Metrics="accuracy", ...
    Shuffle="every-epoch", ...
    OutputNetwork="best-validation", ...
    Plots="training-progress", ...
    Verbose=false);
netCls = trainnet(XTrain,labelsTrain,layersCls,"crossentropy",optionsCls);
%%
%[text] ## Test the Classifier
%[text] The test set contains three recordings per class that the network has never seen, made in separate sessions.
accuracyCls = testnet(netCls,XTest,labelsTest,"accuracy",InputDataFormats="TCB")
predictedCls = onehotdecode(minibatchpredict(netCls,XTest,InputDataFormats="TCB"),classNames,2);
confusionchart(forDisplay(labelsTest),forDisplay(predictedCls),Title="1-D CNN classifier (test set)")
%[text] The CNN also classifies the boundary windows that the threshold got wrong. With only three test recordings per class, treat the exact percentage with care: the result shows that the classes are well separated in new sessions, not that the model will never make a mistake.
%%
%[text] ## Train the Autoencoder on NORMAL Windows
%[text] The autoencoder learns the 24 motion indicators of an undisturbed board and reconstructs them through a 6-neuron bottleneck. Any vibration makes the indicators unfamiliar and the reconstruction error large.
indicatorsOf = @(w) (cell2mat(cellfun(@computeMotionIndicators,w,UniformOutput=false)) - ciMu)./ciSigma;
ZTrain = indicatorsOf(windowsTrain(labelsTrain == "NORMAL"));
ZValNormal = indicatorsOf(windowsVal(labelsVal == "NORMAL"));
ZTest = indicatorsOf(windowsTest);
layersAE = [
    featureInputLayer(24)
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(6)
    reluLayer
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(24)];
optionsAE = trainingOptions("adam", ...
    MaxEpochs=300, ...
    MiniBatchSize=64, ...
    InitialLearnRate=2e-3, ...
    ValidationData={ZValNormal,ZValNormal}, ...
    Shuffle="every-epoch", ...
    OutputNetwork="best-validation", ...
    Plots="training-progress", ...
    Verbose=false);
netAE = trainnet(ZTrain,ZTrain,layersAE,"mse",optionsAE);
%%
%[text] ## Choose a Robust Anomaly Threshold
%[text] Look at the reconstruction error of the `NORMAL` validation windows. Most are tiny, but a few are very large.
scoreValNormal = autoencoderScore(netAE,ZValNormal);
prctile(scoreValNormal,[50 90 95 99])
%[text] A percentile threshold would be pulled up by these few windows. Instead, use a robust rule on the logarithm of the score: the median plus five scaled median absolute deviations (MAD), which ignores a handful of outliers.
logScore = log10(scoreValNormal);
aeThreshold = 10^(median(logScore) + 5*1.4826*mad(logScore,1))
%%
%[text] ## Find What the Autoencoder Found
%[text] Apply the threshold to the test set. The autoencoder flags every tapping and shaking window, as expected, but also a few windows inside `NORMAL` recordings.
scoreTest = autoencoderScore(netAE,ZTest);
flaggedByAE = groupsummary(table(labelsTest,scoreTest > aeThreshold,VariableNames=["Class" "Flagged"]),"Class","mean","Flagged")
%[text] Plot the flagged and unflagged `NORMAL` test windows side by side.
windowsTestNormal = windowsTest(labelsTest == "NORMAL");
scoreTestNormal = scoreTest(labelsTest == "NORMAL");
[~,iFlagged] = max(scoreTestNormal);
[~,iQuiet] = min(scoreTestNormal);
t = (0:99)/100;
tiledlayout(1,2,TileSpacing="compact")
nexttile
x = windowsTestNormal{iQuiet}; plot(t,x(:,1:3) - mean(x(:,1:3)))
ylim([-0.04 0.04]); xlabel("Time (s)"); ylabel("Acceleration (g)")
title(sprintf("NORMAL window, score %.2f",scoreTestNormal(iQuiet)))
nexttile
x = windowsTestNormal{iFlagged}; plot(t,x(:,1:3) - mean(x(:,1:3)))
ylim([-0.04 0.04]); xlabel("Time (s)")
title(sprintf("Flagged NORMAL window, score %.0f",scoreTestNormal(iFlagged)))
legend(["X" "Y" "Z"])
%[text] The flagged windows contain small real bumps, for example someone touching the table during a `NORMAL` recording. The labels describe the whole 30 s recording, but the autoencoder judges each second on its own and notices them. Compare the peak acceleration of the two groups:
peakOf = @(w) cellfun(@(x) max(abs(x(:,1:3) - mean(x(:,1:3))),[],"all"),w);
peakNormal = peakOf(windowsTestNormal);
isFlagged = scoreTestNormal > aeThreshold;
peakSummary = table(["Not flagged"; "Flagged"],[sum(~isFlagged); sum(isFlagged)], ...
    [median(peakNormal(~isFlagged)); median(peakNormal(isFlagged))], ...
    VariableNames=["NORMAL test windows" "Count" "Median peak (g)"])
%[text] This is the value of the hybrid design on your own data: the classifier, trained on recording labels, calls these windows level 0, while the autoencoder reports them as an unrecognized minor vibration. On the board this appears as a separate state.
%%
%[text] ## Visualize the Network Architectures
%[text] The helper `plotNetworkArchitecture` draws each layer as a box whose color shows the layer type and whose height grows with the number of channels. The output size of each layer is printed below it. The classifier reads the raw 100 × 6 window of accelerometer and gyroscope samples, halves the time axis twice with max pooling, and collapses it with global average pooling, so the fully connected layers see a single 32-element summary of the whole second. The autoencoder squeezes the 24 motion indicators through a 6-neuron bottleneck.
plotNetworkArchitecture(netCls,[100 6],"Classifier: 1-D CNN on the 100 × 6 IMU window");
plotNetworkArchitecture(netAE,24,"Autoencoder on 24 motion indicators");
%[text] You can also inspect the networks interactively with `analyzeNetwork(netCls)`, or convert them to Simulink. `exportNetworkToSimulink` creates a model with one Simulink block per layer, which you can simulate with signals from other Simulink blocks. With `FrameBased=true`, the model takes a whole window at each time step, which the max pooling layers need.
if ~isempty(ver("simulink")) && license("test","Simulink")   % Simulink is optional
    exportNetworkToSimulink(netCls,ModelName="vibrationSeverityClassifier",FrameBased=true);
end
%%
%[text] ## Save the Trained Networks
save(fullfile(projectFolder,"models","baseline.mat"),"netCls","netAE","aeThreshold","accuracyCls","accuracyBaseline")
%[text] Next, open **Part3\_CompressClassifier**.

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
