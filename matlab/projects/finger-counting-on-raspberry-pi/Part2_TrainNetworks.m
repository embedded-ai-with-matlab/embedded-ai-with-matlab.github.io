%[text] # Finger Counting on a Raspberry Pi 5
%[text] ## Part 2: Train the Network
%[text] This script trains a small **convolutional neural network (CNN)** that looks at a 96 × 96 color image of a hand and predicts the number of fingers, from 0 to 5. Run **Part1\_PrepareData** first.
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
toSingle = @(x) single(x)/255;      % pixel values between 0 and 1, as preprocessFrame produces on the Pi
XTrain = toSingle(XTrain); XVal = toSingle(XVal); XTest = toSingle(XTest);
%%
%[text] ## Define the Network
%[text] The network has four convolution blocks. Each one finds simple patterns (edges, then finger shapes), and each pooling layer halves the width and height. Global average pooling collapses the last feature maps, so the prediction does not depend on exactly *where* in the image the hand is. Batch normalization steadies the training, and dropout randomly switches off some neurons during training, which helps when there are only about 1,200 training images.
layers = [
    imageInputLayer([96 96 3],Normalization="none")
    convolution2dLayer(3,32,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,Stride=2)
    convolution2dLayer(3,64,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,Stride=2)
    convolution2dLayer(3,128,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,Stride=2)
    convolution2dLayer(3,128,Padding="same")
    batchNormalizationLayer
    reluLayer
    globalAveragePooling2dLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(6)
    softmaxLayer];
plot(dlnetwork(layers))
title("Finger-counting CNN")
%%
%[text] ## Augment the Training Images
%[text] A photo of a hand in a new room will differ from the training photos: the hand is a little rotated, closer, or off-center. Random rotation, scaling, and shifting of the training images teach the network to cope with this. (Mirroring is left out: it did not help in our tests.) The validation and test images are not augmented.
augmenter = imageDataAugmenter( ...
    RandRotation=[-15 15], ...
    RandScale=[0.9 1.1], ...
    RandXTranslation=[-8 8], ...
    RandYTranslation=[-8 8]);
dsTrain = augmentedImageDatastore([96 96 3],XTrain,labelsTrain,DataAugmentation=augmenter);
sample = preview(augmentedImageDatastore([96 96 3],XTrain(:,:,:,1:8),labelsTrain(1:8),DataAugmentation=augmenter));
tiledlayout(2,8,TileSpacing="none",Padding="compact")
for k = 1:8
    nexttile; image(XTrain(:,:,:,k)); axis image off
    nexttile; image(sample.input{k}); axis image off
end
%[text] Top row: the original images; bottom row: one random augmentation of each.
%%
%[text] ## Train
%[text] Training opens the **training-progress window**. The training curves are computed on every mini-batch; the validation curves are computed on the validation groups, which the network never trains on. `OutputNetwork="best-validation"` returns the network from the moment with the lowest validation loss.
%[text] On a laptop CPU, the 80 epochs take a few minutes. **Expect an unsteady validation curve.** The training accuracy reaches nearly 100%, but the validation accuracy jumps between about 25% and 70% from one check to the next: the validation set is small, and its hands come from recording groups that the network has never seen. A second run with the same seed can give a different network.
options = trainingOptions("adam", ...
    MaxEpochs=80, ...
    MiniBatchSize=32, ...
    InitialLearnRate=2e-3, ...
    ValidationData={XVal,labelsVal}, ...
    ValidationFrequency=20, ...
    Metrics="accuracy", ...
    Shuffle="every-epoch", ...
    OutputNetwork="best-validation", ...
    Plots="training-progress", ...
    Verbose=false);
[net,info] = trainnet(dsTrain,layers,"crossentropy",options);
%%
%[text] ## Test the Network
%[text] The test set contains hands from two recording groups that the network has never seen.
accuracyTest = testnet(net,XTest,labelsTest,"accuracy")
predicted = onehotdecode(minibatchpredict(net,XTest),classNames,2);
confusionchart(labelsTest,predicted,Title="Test set (groups never seen in training)")
%[text] The test set has far more fists than other classes, so also look at the accuracy of each class separately. **Treat these numbers with care.** The test set has only 247 hands from two recording groups, and the same network trained with another random seed can score anywhere between 40% and 70% on it: a small dataset gives an unstable estimate. Part 6 tests the model on the Pi's own camera, which is the number that matters.
recallPerClass = array2table(100*diag(confusionmat(labelsTest,predicted))'./countcats(labelsTest)',VariableNames="fingers_" + classNames)
accuracyVal = testnet(net,XVal,labelsVal,"accuracy")
%%
%[text] ## See What the Network Looks At
%[text] **Grad-CAM** highlights the pixels that raised the score of the predicted class. If the network has learned to count fingers, the bright areas should lie on the fingers, not on the face or the wall.
tiledlayout(2,6,TileSpacing="compact",Padding="compact")
for c = 1:6
    idx = find(labelsTest == classNames{c} & predicted == classNames{c},1);
    if isempty(idx), idx = find(labelsTest == classNames{c},1); end
    map = gradCAM(net,XTest(:,:,:,idx),predicted(idx));
    nexttile
    image(XTest(:,:,:,idx)); axis image off
    title(classNames{c} + " fingers")
    nexttile
    image(XTest(:,:,:,idx)); hold on
    imagesc(map,AlphaData=0.5); colormap jet; axis image off
end
%%
%[text] ## Save the Trained Network
save(fullfile(projectFolder,"models","baseline.mat"),"net","accuracyTest","accuracyVal")
%[text] Next, open **Part3\_CompressClassifier**.

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
