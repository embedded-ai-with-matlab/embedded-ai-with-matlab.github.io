%[text] # Digit Recognition on a Raspberry Pi 5
%[text] ## Part 2: Train the Network
%[text] This script trains a small **convolutional neural network (CNN)** that looks at a 28 × 28 image of a digit and predicts which of the ten digits it is. Run **Part1\_PrepareData** first.
%%
%[text] ## Load the Prepared Data
projectFolder = fileparts(mfilename("fullpath"));
if ~isfolder(fullfile(projectFolder,"helpers"))   % the Live Editor can run a temporary copy of the script
    projectFolder = fileparts(matlab.desktop.editor.getActiveFilename);
end
cd(projectFolder)
addpath(fullfile(projectFolder,"helpers"))
load(fullfile(projectFolder,"data","preparedData.mat"))
rng(0)
%%
%[text] ## Define the Network
%[text] The network has three convolution blocks. Each one finds simple patterns (strokes, then curves and corners), and each pooling layer halves the width and height. Global average pooling collapses the last feature maps, and a fully connected layer turns them into ten scores. Batch normalization steadies the training, and dropout switches off some neurons at random during training.
layers = [
    imageInputLayer([28 28 1],Normalization="none")
    convolution2dLayer(3,16,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,Stride=2)
    convolution2dLayer(3,32,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,Stride=2)
    convolution2dLayer(3,64,Padding="same")
    batchNormalizationLayer
    reluLayer
    globalAveragePooling2dLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(10)
    softmaxLayer];
figure
plot(dlnetwork(layers))
title("Digit-recognition CNN")
%%
%[text] ## Augment the Training Images
%[text] A digit that you write on paper and hold in front of a camera is a little rotated, slanted, larger or smaller, and not perfectly centered. Random rotation, shearing, scaling, and shifting of the training images teach the network to cope with this. The validation and test images are not augmented.
augmenter = imageDataAugmenter( ...
    RandRotation=[-12 12], ...
    RandXShear=[-12 12], ...
    RandScale=[0.8 1.2], ...
    RandXTranslation=[-3 3], ...
    RandYTranslation=[-3 3]);
dsTrain = augmentedImageDatastore([28 28 1],XTrain,labelsTrain,DataAugmentation=augmenter);
sample = preview(augmentedImageDatastore([28 28 1],XTrain(:,:,:,1:8),labelsTrain(1:8),DataAugmentation=augmenter));
figure
tiledlayout(2,8,TileSpacing="none",Padding="compact")
for k = 1:8
    nexttile; imshow(XTrain(:,:,:,k))
    nexttile; imshow(sample.input{k})
end
%[text] Top row: the original images; bottom row: one random augmentation of each.
%%
%[text] ## Train
%[text] Training opens the **training-progress window**. The training curves are computed on every mini-batch; the validation curves are computed on the validation images, which the network never trains on. `OutputNetwork="best-validation"` returns the network from the check with the lowest validation loss.
options = trainingOptions("adam", ...
    MaxEpochs=30, ...
    MiniBatchSize=64, ...
    InitialLearnRate=2e-3, ...
    ValidationData={XVal,labelsVal}, ...
    ValidationFrequency=50, ...
    Metrics="accuracy", ...
    Shuffle="every-epoch", ...
    OutputNetwork="best-validation", ...
    Plots="training-progress", ...
    Verbose=false);
[net,info] = trainnet(dsTrain,layers,"crossentropy",options);
%%
%[text] ## Test the Network
accuracyTest = testnet(net,XTest,labelsTest,"accuracy")
predicted = onehotdecode(minibatchpredict(net,XTest),classNames,2);
figure
confusionchart(labelsTest,predicted,Title="Test set")
recallPerClass = array2table(100*diag(confusionmat(labelsTest,predicted))'./countcats(labelsTest)',VariableNames="digit_" + classNames)
accuracyVal = testnet(net,XVal,labelsVal,"accuracy")
%[text] The test accuracy is high because the test images are written in the same styles as the training images. It is not a promise about your handwriting, your pen, and the camera: Part 6 measures that.
%%
%[text] ## See What the Network Looks At
%[text] **Grad-CAM** highlights the pixels that raised the score of the predicted class.
figure
tiledlayout(2,10,TileSpacing="compact",Padding="compact")
for c = 1:10
    idx = find(labelsTest == classNames{c} & predicted == classNames{c},1);
    map = gradCAM(net,XTest(:,:,:,idx),predicted(idx));
    nexttile; imshow(XTest(:,:,:,idx)); title(classNames{c})
    ax = nexttile; imshow(XTest(:,:,:,idx)); hold on; imagesc(map,AlphaData=0.5); colormap(ax,"jet")
end
%%
%[text] ## Save the Trained Network
save(fullfile(projectFolder,"models","baseline.mat"),"net","accuracyTest","accuracyVal")
%[text] Next, open **Part3\_CompressNetwork**.

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
