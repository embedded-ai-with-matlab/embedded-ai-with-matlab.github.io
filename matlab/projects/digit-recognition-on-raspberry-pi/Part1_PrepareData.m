%[text] # Digit Recognition on a Raspberry Pi 5
%[text] ## Part 1: Prepare the Digit Images
%[text] A **Raspberry Pi 5** with a camera reads a handwritten digit from 0 to 9: you write a digit on a sheet of paper, hold it in front of the camera, and the Pi reports which digit it is. The network runs on the Pi as generated C++ code.
%[text] The workflow is the same as in the finger-counting tutorial, with a simpler task and cleaner data:
%[text] - **Part 1 — Prepare data** (this script)
%[text] - **Part 2 — Train the network**
%[text] - **Part 3 — Compress the network**
%[text] - **Part 4 — Quantize the network**
%[text] - **Part 5 — Generate code**
%[text] - **Part 6 — Deploy to the Raspberry Pi and run on the camera** \
%[text] The training images are the **digit data set that comes with Deep Learning Toolbox**: 10,000 grayscale images of handwritten digits, 28 × 28 pixels, with white digits on a black background. No download is needed.
%%
%[text] ## Set Up the Project
projectFolder = fileparts(mfilename("fullpath"));
if ~isfolder(fullfile(projectFolder,"helpers"))   % the Live Editor can run a temporary copy of the script
    projectFolder = fileparts(matlab.desktop.editor.getActiveFilename);
end
cd(projectFolder)
addpath(fullfile(projectFolder,"helpers"))
rng(0)
%%
%[text] ## Load the Images
%[text] An `imageDatastore` finds the images in the ten folders, and the label comes from the folder name.
dataFolder = fullfile(matlabroot,"toolbox","nnet","nndemos","nndatasets","DigitDataset");
imds = imageDatastore(dataFolder,IncludeSubfolders=true,FileExtensions=".png",LabelSource="foldernames");
classNames = categories(imds.Labels)
countEachLabel(imds)
%%
%[text] ## Look at the Images
%[text] Plot ten random images for each digit.
tiledlayout(10,10,TileSpacing="none",Padding="compact")
for c = 1:10
    idx = find(imds.Labels == classNames{c});
    idx = idx(randperm(numel(idx),10));
    for k = 1:10
        nexttile
        imshow(readimage(imds,idx(k)))
    end
end
%[text] The writing styles differ a lot: thin and thick strokes, slanted and upright digits, and some digits that look alike (1 and 7, 4 and 9). The digits fill most of the 28 × 28 image and are centered.
%%
%[text] ## Check the Image Properties
sample = readall(subset(imds,randperm(numel(imds.Files),500)));
sizes = unique(cell2mat(cellfun(@(x) [size(x,1) size(x,2) size(x,3)],sample,UniformOutput=false)),"rows")
pixelClass = unique(cellfun(@class,sample,UniformOutput=false))
brightPixels = mean(cellfun(@(x) mean(x(:) > 128),sample))
%[text] All images are 28 × 28 pixels with one channel and 8-bit pixels. Only a small part of the pixels belongs to the digit.
%%
%[text] ## Split the Data
%[text] Use 70% of the images for training, 15% for validation, and 15% for testing, with the same share of every digit in each set. The data set does not say who wrote each digit, so writers can appear in more than one set, and the test accuracy will be a little better than on a stranger's handwriting. Part 6 tests the network on your own writing.
[imdsTrain,imdsVal,imdsTest] = splitEachLabel(imds,0.7,0.15,"randomized");
splitCounts = table(["Training";"Validation";"Test"],[numel(imdsTrain.Files);numel(imdsVal.Files);numel(imdsTest.Files)],VariableNames=["Set" "Images"])
%%
%[text] ## Load the Images into Arrays
%[text] The images are small, so keep them all in memory as one array, with pixel values between 0 and 1.
XTrain = single(readImages(imdsTrain))/255; labelsTrain = imdsTrain.Labels;
XVal = single(readImages(imdsVal))/255; labelsVal = imdsVal.Labels;
XTest = single(readImages(imdsTest))/255; labelsTest = imdsTest.Labels;
size(XTrain)
%%
%[text] ## Save the Prepared Data
save(fullfile(projectFolder,"data","preparedData.mat"), ...
    "XTrain","labelsTrain","XVal","labelsVal","XTest","labelsTest","classNames")
%[text] Next, open **Part2\_TrainNetwork**.
%%
%[text] ## Helper Function
function X = readImages(ds)
images = readall(ds);
X = cat(4,images{:});
end

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
