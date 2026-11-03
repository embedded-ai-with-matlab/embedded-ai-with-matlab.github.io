%[text] # Part 4: Quantize Both Networks to 8-Bit Integers
%[text] Quantization stores weights and activations as 8-bit integers with a scaling factor per tensor, which cuts parameter memory by about 4x and lets the microcontroller use integer arithmetic. The workflow uses a `dlquantizer` object: `prepareNetwork`, then `calibrate` with representative data, then `quantize`.
%[text] Run **Part3\_CompressClassifier** first.
%%
%[text] ## Load the Data and Networks
projectFolder = fileparts(mfilename("fullpath"));
cd(projectFolder)
addpath(fullfile(projectFolder,"helpers"),fullfile(projectFolder,"codegen"))
load(fullfile(projectFolder,"data","preparedData.mat"))
load(fullfile(projectFolder,"models","baseline.mat"),"netCls","netAE")
load(fullfile(projectFolder,"models","compressed.mat"),"netClsCompressed")
rng(0)

prepare = @(w) cellfun(@(x) (x - mean(x,1))./channelScale,w,UniformOutput=false);
XTrain = prepare(windowsTrain);
XTest = prepare(windowsTest);
indicatorsOf = @(w) (cell2mat(cellfun(@computeMotionIndicators,w,UniformOutput=false)) - ciMu)./ciSigma;
ZTrain = indicatorsOf(windowsTrain(labelsTrain == "NORMAL"));
ZValNormal = indicatorsOf(windowsVal(labelsVal == "NORMAL"));
ZTest = indicatorsOf(windowsTest);
%%
%[text] ## Quantize the Compressed Classifier
%[text] Calibrate with a third of the training windows from all three classes, so that the recorded dynamic ranges cover everything from a still board to strong shaking.
quantObjCls = dlquantizer(netClsCompressed,ExecutionEnvironment="MATLAB");
prepareNetwork(quantObjCls);
calibrate(quantObjCls,arrayDatastore(XTrain(1:3:end),OutputType="same"));
netClsQ = quantize(quantObjCls,ExponentScheme="histogram");
accuracyClsQ = testnet(netClsQ,XTest,labelsTest,"accuracy",InputDataFormats="TCB")
%%
%[text] ## Quantize the Autoencoder
quantObjAE = dlquantizer(netAE,ExecutionEnvironment="MATLAB");
prepareNetwork(quantObjAE);
calibrate(quantObjAE,arrayDatastore(ZTrain,IterationDimension=1));
netAEQ = quantize(quantObjAE,ExponentScheme="histogram");
%[text] Recompute the robust threshold with the quantized autoencoder.
logScore = log10(autoencoderScore(netAEQ,ZValNormal));
aeThreshold = 10^(median(logScore) + 5*1.4826*mad(logScore,1))
flaggedByAEQ = groupsummary(table(labelsTest,autoencoderScore(netAEQ,ZTest) > aeThreshold, ...
    VariableNames=["Class" "Flagged"]),"Class","mean","Flagged")
%%
%[text] ## Compare Memory and Accuracy Across the Workflow
summaries = [networkSummary(netCls); networkSummary(netClsCompressed); networkSummary(netClsQ)];
accuracyOriginal = testnet(netCls,XTest,labelsTest,"accuracy",InputDataFormats="TCB");
accuracyCompressed = testnet(netClsCompressed,XTest,labelsTest,"accuracy",InputDataFormats="TCB");
stageNames = ["Original (float32)" "Pruned + projected (float32)" "Pruned + projected + quantized (int8)"];
stage = categorical(stageNames,stageNames)';
memoryKB = [summaries.MemoryKB]';
results = table(stage,[summaries.NumLearnables]',memoryKB,[accuracyOriginal; accuracyCompressed; accuracyClsQ], ...
    VariableNames=["Stage" "Learnables" "Parameter memory (KB)" "Test accuracy (%)"])
%%
bar(stage,memoryKB)
ylabel("Parameter memory (KB)")
title(sprintf("Classifier memory reduced by %.0f%%",100*(1 - memoryKB(end)/memoryKB(1))))
%%
%[text] ## Save the Deployment Files
save(fullfile(projectFolder,"models","netClsQ.mat"),"netClsQ")
save(fullfile(projectFolder,"models","netAEQ.mat"),"netAEQ")
save(fullfile(projectFolder,"models","deploymentParams.mat"),"channelScale","ciMu","ciSigma","aeThreshold")
save(fullfile(projectFolder,"models","quantizationResults.mat"),"results")
%[text] Next, open **Part5\_GenerateCode**.

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
