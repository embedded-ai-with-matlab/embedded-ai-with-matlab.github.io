%[text] # Part 3: Compress the Classifier with Pruning and Projection
%[text] The vibration severity classifier is small already, but every kilobyte saved on the microcontroller leaves room for other application code and shortens inference. This script shrinks the 1-D CNN with the two **structural** compression techniques in Deep Learning Toolbox Model Compression Library, in the recommended order: *first prune, then project, then quantize* (Part 4), fine-tuning after each step to recover accuracy.
%[text] - **Taylor pruning** removes whole convolution filters and neurons whose removal changes the loss least, estimated with a first-order Taylor expansion.
%[text] - **Projection** uses principal component analysis (PCA) of each layer's activations to replace a large layer with two smaller ones that span the most important directions. \
%[text] Run **Part2\_TrainNetworks** first.
%%
%[text] ## Load the Data and the Trained Classifier
projectFolder = fileparts(mfilename("fullpath"));
cd(projectFolder)
addpath(fullfile(projectFolder,"helpers"),fullfile(projectFolder,"codegen"))
load(fullfile(projectFolder,"data","preparedData.mat"))
load(fullfile(projectFolder,"models","baseline.mat"),"netCls")
rng(0)

prepare = @(w) cellfun(@(x) (x - mean(x,1))./channelScale,w,UniformOutput=false);
XTrain = prepare(windowsTrain);  TTrain = labelsTrain;
XVal   = prepare(windowsVal);    TVal   = labelsVal;
XTest  = prepare(windowsTest);   TTest  = labelsTest;
%%
%[text] ## Analyze the Original Network
estimateNetworkMetrics(netCls)
summaryOriginal = networkSummary(netCls);
accuracyOriginal = testnet(netCls,XTest,TTest,"accuracy",InputDataFormats="TCB");
%%
%[text] ## Set Up Fine-Tuning Options
optionsPruning = trainingOptions("adam", ...
    MaxEpochs=10, ...
    MiniBatchSize=64, ...
    InitialLearnRate=2e-3, ...
    InputDataFormats="TCB", ...
    ValidationData={XVal,TVal}, ...
    Metrics="accuracy", ...
    Shuffle="every-epoch", ...
    Plots="none", ...
    Verbose=false);
optionsFineTune = trainingOptions("adam", ...
    MaxEpochs=30, ...
    MiniBatchSize=64, ...
    InitialLearnRate=5e-4, ...
    InputDataFormats="TCB", ...
    ValidationData={XVal,TVal}, ...
    Metrics="accuracy", ...
    Shuffle="every-epoch", ...
    OutputNetwork="best-validation", ...
    Plots="training-progress", ...
    Verbose=false);
%%
%[text] ## Step 1: Taylor Pruning
%[text] Remove 50% of the learnables in steps of 5%. The classes in this dataset are well separated, so the network tolerates more pruning than the ceiling-fan classifier in the first tutorial.
[netPruned,infoPruned] = compressNetworkUsingTaylorPruning(netCls,XTrain,TTrain,"crossentropy",optionsPruning, ...
    LearnablesReductionGoal=0.5, ...
    LearnablesReductionIncrement=0.05);
infoPruned.PrunedLayers
%%
%[text] Compare the number of filters in each convolution layer before and after pruning.
convNames = ["conv1d_1" "conv1d_2" "conv1d_3"];
filtersBefore = arrayfun(@(n) getLayer(netCls,n).NumFilters,convNames);
filtersAfter = arrayfun(@(n) getLayer(netPruned,n).NumFilters,convNames);
bar(categorical(convNames),[filtersBefore; filtersAfter]')
legend(["Original" "Pruned"])
ylabel("Number of filters")
title("Filters removed by Taylor pruning")
%%
%[text] ## Fine-Tune the Pruned Network
netPruned = trainnet(XTrain,TTrain,netPruned,"crossentropy",optionsFineTune);
summaryPruned = networkSummary(netPruned);
accuracyPruned = testnet(netPruned,XTest,TTest,"accuracy",InputDataFormats="TCB")
%%
%[text] ## Step 2: Projection
%[text] Reduce the learnables of the pruned network by a further 50% with PCA-based projection.
[netProjected,infoProjected] = compressNetworkUsingProjection(netPruned,XTrain, ...
    InputDataFormats="TCB", ...
    LearnablesReductionGoal=0.5);
accuracyProjectedBeforeFineTune = testnet(netProjected,XTest,TTest,"accuracy",InputDataFormats="TCB")
%%
%[text] ## Fine-Tune the Projected Network
netProjected = trainnet(XTrain,TTrain,netProjected,"crossentropy",optionsFineTune);
summaryProjected = networkSummary(netProjected);
accuracyProjected = testnet(netProjected,XTest,TTest,"accuracy",InputDataFormats="TCB")
%%
%[text] ## Compare the Compression Stages
stage = categorical(["Original" "Pruned" "Pruned + projected"],["Original" "Pruned" "Pruned + projected"])';
learnables = [summaryOriginal.NumLearnables; summaryPruned.NumLearnables; summaryProjected.NumLearnables];
memoryKB = [summaryOriginal.MemoryKB; summaryPruned.MemoryKB; summaryProjected.MemoryKB];
accuracy = [accuracyOriginal; accuracyPruned; accuracyProjected];
compressionResults = table(stage,learnables,memoryKB,accuracy, ...
    VariableNames=["Stage" "Learnables" "Memory (KB)" "Test accuracy (%)"])
%%
tiledlayout(1,2)
nexttile
bar(stage,learnables)
ylabel("Learnable parameters")
title("Network size")
nexttile
bar(stage,accuracy)
ylim([50 100])
ylabel("Test accuracy (%)")
title("Accuracy")
%%
%[text] ## Save the Compressed Classifier
netClsCompressed = netProjected;
save(fullfile(projectFolder,"models","compressed.mat"),"netClsCompressed","compressionResults")
%[text] Next, open **Part4\_QuantizeNetworks**.

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
