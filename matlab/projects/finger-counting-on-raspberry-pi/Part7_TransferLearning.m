%[text] # Finger Counting on a Raspberry Pi 5
%[text] ## Part 7 (continuation): Improve the Result with Transfer Learning
%[text] The small CNN of Part 2 learns from about 1,200 training hands and ends at 59% accuracy on the test groups. A network that was **already trained on a million photos** has learned edges, textures, and shapes that also help to see fingers. **Transfer learning** starts from such a network, replaces its last layer with one for six classes, and trains it on your hands. This script uses **SqueezeNet**, a small network that comes with Deep Learning Toolbox. Run **Part1\_PrepareData** first; **Part2\_TrainNetworks** is only needed for the comparison.
%%
%[text] ## Load the Data and the Pretrained Network
projectFolder = fileparts(mfilename("fullpath"));
if ~isfolder(fullfile(projectFolder,"helpers"))   % the Live Editor can run a temporary copy of the script
    projectFolder = fileparts(matlab.desktop.editor.getActiveFilename);
end
cd(projectFolder)
addpath(fullfile(projectFolder,"helpers"))
load(fullfile(projectFolder,"data","preparedData.mat"))
%[text] SqueezeNet expects pixel values from 0 to 255, so, unlike in Part 2, the images are not divided by 255. `imagePretrainedNetwork` returns the network with its last learnable layer already replaced by one with six outputs. SqueezeNet was trained on 227 × 227 images, but it is fully convolutional, so it also works on the 96 × 96 hand crops. Replacing the input layer sets the size.
XTrain = single(XTrain); XVal = single(XVal); XTest = single(XTest);
net0 = imagePretrainedNetwork("squeezenet",NumClasses=6);
inputLayer = net0.Layers(1);
net0 = dlnetwork(replaceLayer(layerGraph(net0),inputLayer.Name, ...
    imageInputLayer([96 96 3],Normalization="zerocenter",Mean=inputLayer.Mean,Name=inputLayer.Name)));
networkSummary(net0)
%[text] SqueezeNet is built from **fire modules**: a 1 × 1 convolution squeezes the channels, and a 1 × 1 and a 3 × 3 convolution expand them again, side by side, and a concatenation layer joins the results. This design gives the network fewer parameters than a plain CNN of similar power. You can inspect the layers of the network interactively with `analyzeNetwork(net0)`.
%%
%[text] ## Train
%[text] Fine-tune the whole network with the same augmentation as in Part 2. The learning rate is small (1e-4), because the pretrained weights are already good and should only be adjusted.
augmenter = imageDataAugmenter(RandRotation=[-15 15],RandScale=[0.9 1.1],RandXTranslation=[-8 8],RandYTranslation=[-8 8]);
dsTrain = augmentedImageDatastore([96 96 3],XTrain,labelsTrain,DataAugmentation=augmenter);
dsVal = augmentedImageDatastore([96 96 3],XVal,labelsVal);
options = trainingOptions("adam", ...
    MaxEpochs=12, ...
    MiniBatchSize=32, ...
    InitialLearnRate=1e-4, ...
    ValidationData=dsVal, ...
    ValidationFrequency=20, ...
    Metrics="accuracy", ...
    Shuffle="every-epoch", ...
    OutputNetwork="best-validation", ...
    Plots="training-progress", ...
    Verbose=false);
rng(1)
[netTransfer,info] = trainnet(dsTrain,net0,"crossentropy",options);
%[text] Training takes about a minute on a laptop CPU.
%%
%[text] ## Test and Compare
accuracyOf = @(n) 100*mean(onehotdecode(double(minibatchpredict(n,XTest)),classNames,2) == labelsTest);
accuracyTransfer = accuracyOf(netTransfer)
predicted = onehotdecode(minibatchpredict(netTransfer,XTest),classNames,2);
figure
confusionchart(labelsTest,predicted,Title="SqueezeNet, test set (groups never seen in training)")
recallTransfer = 100*diag(confusionmat(labelsTest,predicted))'./countcats(labelsTest)';
%[text] Compare with the small CNN of Part 2, on the same 247 test hands:
baseline = load(fullfile(projectFolder,"models","baseline.mat"),"net");
predictedBaseline = onehotdecode(minibatchpredict(baseline.net,XTest/255),classNames,2);
recallBaseline = 100*diag(confusionmat(labelsTest,predictedBaseline))'./countcats(labelsTest)';
accuracyBaseline = 100*mean(predictedBaseline == labelsTest)
figure
bar(categorical(classNames,classNames),[recallBaseline; recallTransfer]')
legend(["Small CNN from scratch" "SqueezeNet, transfer learning"],Location="southeast")
xlabel("Number of fingers"), ylabel("Test accuracy of the class (%)"), ylim([0 105])
title("Accuracy of each class on the test set"), grid on
%%
%[text] ## How Much Does the Seed Matter?
%[text] Part 2 showed that the small CNN scores anywhere between about 40% and 70%, depending on the random seed. Train the transfer-learning network with two more seeds to see whether it is steadier.
seeds = [1 2 3];
accuracySeeds = zeros(size(seeds));
accuracySeeds(1) = accuracyTransfer;
for k = 2:numel(seeds)
    rng(seeds(k))
    netSeed = trainnet(dsTrain,net0,"crossentropy",options);
    accuracySeeds(k) = accuracyOf(netSeed);
end
seedResults = table(seeds',accuracySeeds',VariableNames=["Seed" "TestAccuracy"])
meanAccuracy = mean(accuracySeeds)
%%
%[text] ## See What the Network Looks At
%[text] Grad-CAM for the first correctly classified test hand of each class:
figure
tiledlayout(2,6,TileSpacing="compact",Padding="compact")
for c = 1:6
    idx = find(labelsTest == classNames{c} & predicted == classNames{c},1);
    if isempty(idx), idx = find(labelsTest == classNames{c},1); end
    map = gradCAM(netTransfer,XTest(:,:,:,idx),predicted(idx));
    nexttile
    imshow(uint8(XTest(:,:,:,idx)))
    title(classNames{c} + " fingers")
    ax = nexttile; imshow(uint8(XTest(:,:,:,idx))); hold on
    imagesc(map,AlphaData=0.5); colormap(ax,"jet")
end
%%
%[text] ## Prune the Network
%[text] SqueezeNet has 726,000 learnable parameters, three times as many as the small CNN. Taylor pruning removes the filters that matter least; the fine-tuning afterwards recovers the accuracy. In Part 3, projection was part of the recipe; for SqueezeNet it did not work well (the accuracy fell from 87% to 76% in a test), so only pruning is used here.
optionsPruning = trainingOptions("adam",MaxEpochs=4,MiniBatchSize=32,InitialLearnRate=1e-4,Shuffle="every-epoch",Plots="none",Verbose=false);
optionsFineTune = trainingOptions("adam",MaxEpochs=8,MiniBatchSize=32,InitialLearnRate=5e-5, ...
    ValidationData=dsVal,ValidationFrequency=20,Metrics="accuracy",Shuffle="every-epoch", ...
    OutputNetwork="best-validation",Plots="none",Verbose=false);
rng(1)
[netPruned,infoPruned] = compressNetworkUsingTaylorPruning(netTransfer,XTrain,labelsTrain,"crossentropy",optionsPruning, ...
    LearnablesReductionGoal=0.5,LearnablesReductionIncrement=0.1);
netPruned = trainnet(dsTrain,netPruned,"crossentropy",optionsFineTune);
stage = categorical(["Small CNN (Part 2)" "SqueezeNet" "SqueezeNet, pruned"],["Small CNN (Part 2)" "SqueezeNet" "SqueezeNet, pruned"])';
summaries = [networkSummary(baseline.net); networkSummary(netTransfer); networkSummary(netPruned)];
accuracy = [accuracyBaseline; accuracyTransfer; accuracyOf(netPruned)];
transferResults = table(stage,[summaries.NumLearnables]',[summaries.MemoryKB]',accuracy, ...
    VariableNames=["Network" "Learnables" "Memory (KB)" "Test accuracy (%)"])
%%
%[text] ## Save the Network
save(fullfile(projectFolder,"models","netTransfer.mat"),"netPruned")
save(fullfile(projectFolder,"models","transferResults.mat"),"transferResults","seedResults","recallBaseline","recallTransfer")
%[text] Next, open **Part8\_DeployTransferNetwork**.

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
