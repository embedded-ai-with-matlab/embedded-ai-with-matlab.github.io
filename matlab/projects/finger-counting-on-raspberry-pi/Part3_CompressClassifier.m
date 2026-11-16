%[text] # Finger Counting on a Raspberry Pi 5
%[text] ## Part 3: Compress the Network
%[text] A Raspberry Pi 5 has plenty of memory, but a smaller network still loads faster, uses less memory bandwidth, and runs faster. This script makes the network smaller in two steps and checks what each step costs in accuracy:
%[text] - **Taylor pruning** removes the convolution filters that matter least.
%[text] - **Projection** replaces each layer with two smaller layers that keep its principal directions. \
%[text] Run **Part2\_TrainNetworks** first.
%%
%[text] ## Load the Network and the Data
projectFolder = fileparts(mfilename("fullpath"));
if ~isfolder(fullfile(projectFolder,"helpers"))   % the Live Editor can run a temporary copy of the script
    projectFolder = fileparts(matlab.desktop.editor.getActiveFilename);
end
cd(projectFolder)
addpath(fullfile(projectFolder,"helpers"))
load(fullfile(projectFolder,"data","preparedData.mat"))
load(fullfile(projectFolder,"models","baseline.mat"),"net")
rng(0)
toSingle = @(x) single(x)/255;
XTrain = toSingle(XTrain); XVal = toSingle(XVal); XTest = toSingle(XTest);
%%
%[text] ## Measure the Starting Point
estimateNetworkMetrics(net)
summaryOriginal = networkSummary(net);
accuracyOriginal = testnet(net,XTest,labelsTest,"accuracy");
%%
%[text] ## Define the Training Options
%[text] Pruning and projection both damage the network a little, so each step is followed by **fine-tuning**: a few more epochs of training, with the same augmentation as in Part 2, that let the remaining weights adapt.
augmenter = imageDataAugmenter(RandRotation=[-15 15],RandScale=[0.9 1.1],RandXTranslation=[-8 8],RandYTranslation=[-8 8]);
dsTrain = augmentedImageDatastore([96 96 3],XTrain,labelsTrain,DataAugmentation=augmenter);
optionsPruning = trainingOptions("adam", ...
    MaxEpochs=10, ...
    MiniBatchSize=32, ...
    InitialLearnRate=2e-3, ...
    Shuffle="every-epoch", ...
    Plots="none", ...
    Verbose=false);
optionsFineTune = trainingOptions("adam", ...
    MaxEpochs=30, ...
    MiniBatchSize=32, ...
    InitialLearnRate=5e-4, ...
    ValidationData={XVal,labelsVal}, ...
    ValidationFrequency=20, ...
    Metrics="accuracy", ...
    Shuffle="every-epoch", ...
    OutputNetwork="best-validation", ...
    Plots="training-progress", ...
    Verbose=false);
%%
%[text] ## Prune the Network
%[text] Taylor pruning scores every filter by how much removing it would change the loss, and removes the least important ones in small steps. Remove 50% of the learnable parameters in steps of 5%.
[netPruned,infoPruned] = compressNetworkUsingTaylorPruning(net,XTrain,labelsTrain,"crossentropy",optionsPruning, ...
    LearnablesReductionGoal=0.5, ...
    LearnablesReductionIncrement=0.05);
infoPruned.PrunedLayers
%[text] Compare the number of filters in each convolution layer before and after.
layerNames = string({net.Layers.Name});
convNames = layerNames(startsWith(layerNames,"conv_"));
filtersBefore = arrayfun(@(n) getLayer(net,n).NumFilters,convNames);
filtersAfter = arrayfun(@(n) getLayer(netPruned,n).NumFilters,convNames);
figure
bar(categorical(convNames,convNames),[filtersBefore; filtersAfter]')
legend(["Original" "Pruned"])
ylabel("Number of filters")
title("Filters removed by Taylor pruning")
%%
%[text] Fine-tune the pruned network.
netPruned = trainnet(dsTrain,netPruned,"crossentropy",optionsFineTune);
summaryPruned = networkSummary(netPruned);
accuracyPruned = testnet(netPruned,XTest,labelsTest,"accuracy")
%%
%[text] ## Project the Network
%[text] Projection looks at the activations of each layer on the training data and keeps only the directions in which they vary. Remove a further 50% of the learnable parameters, then fine-tune.
[netProjected,infoProjected] = compressNetworkUsingProjection(netPruned,XTrain, ...
    LearnablesReductionGoal=0.5);
accuracyProjectedBeforeFineTune = testnet(netProjected,XTest,labelsTest,"accuracy")
netProjected = trainnet(dsTrain,netProjected,"crossentropy",optionsFineTune);
summaryProjected = networkSummary(netProjected);
accuracyProjected = testnet(netProjected,XTest,labelsTest,"accuracy")
%%
%[text] ## Compare the Three Networks
stage = categorical(["Original" "Pruned" "Pruned + projected"],["Original" "Pruned" "Pruned + projected"])';
learnables = [summaryOriginal.NumLearnables; summaryPruned.NumLearnables; summaryProjected.NumLearnables];
memoryKB = [summaryOriginal.MemoryKB; summaryPruned.MemoryKB; summaryProjected.MemoryKB];
accuracy = [accuracyOriginal; accuracyPruned; accuracyProjected];
compressionResults = table(stage,learnables,memoryKB,accuracy, ...
    VariableNames=["Stage" "Learnables" "Memory (KB)" "Test accuracy (%)"])
figure
tiledlayout(1,2)
nexttile
bar(stage,learnables)
ylabel("Learnable parameters")
title("Network size")
nexttile
bar(stage,accuracy)
ylabel("Test accuracy (%)")
title("Accuracy")
%[text] Do not read too much into the accuracy column. The test set is small (see Part 2), and fine-tuning is also extra training, so a compressed network can even score higher than the original, as the pruned network does here. What the table shows reliably is the reduction in size: the network now needs about a quarter of the parameter memory.
%%
%[text] ## Save the Compressed Network
netCompressed = netProjected;
save(fullfile(projectFolder,"models","compressed.mat"),"netCompressed","compressionResults")
%[text] Next, open **Part4\_QuantizeNetwork**.

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
