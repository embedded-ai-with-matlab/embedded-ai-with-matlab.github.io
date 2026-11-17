%[text] # Finger Counting on a Raspberry Pi 5
%[text] ## Part 1: Prepare the Hand Images
%[text] The first two tutorials ran on a microcontroller and read a motion sensor. This tutorial moves to a **Raspberry Pi 5** and a **camera**. The application is a **finger counter**: the camera sees a hand, and the Pi reports how many fingers are held up, from 0 to 5.
%[text] The workflow is the same as before, applied to images:
%[text] - **Part 1 — Prepare data** (this script)
%[text] - **Part 2 — Train networks**
%[text] - **Part 3 — Compress the classifier**
%[text] - **Part 4 — Quantize the network**
%[text] - **Part 5 — Generate code**
%[text] - **Part 6 — Deploy to the Raspberry Pi and run on the camera** \
%[text] The training photos come from the public **Hands Dataset** (Apache License 2.0). In Part 6, you test the deployed model on the Pi's own camera.
%%
%[text] ## Set Up the Project
projectFolder = fileparts(mfilename("fullpath"));
if ~isfolder(fullfile(projectFolder,"helpers"))   % the Live Editor can run a temporary copy of the script
    projectFolder = fileparts(matlab.desktop.editor.getActiveFilename);   % then use the folder of the open script
end
cd(projectFolder)
addpath(fullfile(projectFolder,"helpers"))
rng(0)
%%
%[text] ## Get the Dataset
%[text] The [Hands Dataset](https://github.com/SubhiH/HandsDataset) contains about 1,550 photos (800 × 600 pixels) taken with a webcam, in different rooms. Each photo has one or more rectangles around hands, and each rectangle is labeled with the number of open fingers (0 to 5). The photos show a person's face, so they are not part of this repository: `downloadHandsDataset` downloads them from GitHub (about 430 MB) the first time.
rawFolder = fullfile(projectFolder,"data","raw");
datasetFolder = downloadHandsDataset(rawFolder);
%%
%[text] ## Read the Labels
%[text] The label files list one hand per row: the photo name, its size, the class, and the corners of the rectangle.
labelsTable = [readtable(fullfile(datasetFolder,"data","train_labels.csv")); ...
    readtable(fullfile(datasetFolder,"data","test_labels.csv"))];
head(labelsTable,4)
labelsTable.Fingers = categorical(extractBefore(string(labelsTable.class),"-"));
classNames = categories(labelsTable.Fingers);
numHands = height(labelsTable)
numPhotos = numel(unique(labelsTable.filename))
countsPerClass = countcats(labelsTable.Fingers)'
%[text] There are about 1,700 labeled hands, and the classes are not balanced: there are twice as many fists (0 fingers) as hands with 1 to 4 fingers.
%%
%[text] ## Look at the Photos
%[text] Plot six photos with their labeled rectangles.
photoFile = @(name) fullfile(datasetFolder,"images",ternary(isfile(fullfile(datasetFolder,"images","train",name)),"train","test"),name);
photos = unique(labelsTable.filename);
tiledlayout(2,3,TileSpacing="compact",Padding="compact")
for k = randperm(numel(photos),6)
    nexttile
    image(imread(photoFile(photos{k}))); axis image off; hold on
    rows = labelsTable(strcmp(labelsTable.filename,photos{k}),:);
    for j = 1:height(rows)
        rectangle(Position=[rows.xmin(j) rows.ymin(j) rows.xmax(j)-rows.xmin(j) rows.ymax(j)-rows.ymin(j)],EdgeColor="g",LineWidth=2)
        text(rows.xmin(j)+4,rows.ymin(j)+25,string(rows.Fingers(j)),Color="g",FontSize=18,FontWeight="bold")
    end
end
%[text] The dataset was made for object detection. A classifier does not need to find the hand, so the next step cuts out each hand.
%%
%[text] ## Cut Out the Hands
%[text] `cropHand` makes each rectangle square, adds a 20% margin so that the fingertips are not cut off, and resizes it to 96 × 96 pixels. The margin and the square shape keep the proportions of the hand.
outSize = 96;
X = zeros(outSize,outSize,3,numHands,"uint8");
for k = 1:numHands
    img = imread(photoFile(labelsTable.filename{k}));
    box = [labelsTable.xmin(k) labelsTable.ymin(k) labelsTable.xmax(k) labelsTable.ymax(k)];
    X(:,:,:,k) = cropHand(img,box,outSize,0.2);
end
labels = labelsTable.Fingers;
size(X)
%[text] Plot eight random hands for each finger count.
tiledlayout(6,8,TileSpacing="none",Padding="compact")
for c = 1:6
    idx = find(labels == classNames{c});
    idx = idx(randperm(numel(idx),8));
    for k = 1:8
        nexttile
        image(X(:,:,:,idx(k))); axis image off
        if k == 1, title(classNames{c} + " fingers",HorizontalAlignment="left"); end
    end
end
%%
%[text] ## Split the Data by Group
%[text] The photo names start with a recording group, for example `996_pic00012.jpg`. Photos of one group show the same room, the same light, and the same person, so they look alike. If photos of one group were in both the training and test sets, the test would be too easy. Instead, **whole groups** go to one set: this gives an honest estimate of how the model behaves in a new room.
%[text] About 100 photos of 5 fingers have names without a group (`pic00005.jpg` or `Hand_0000026.jpg`). They come from a different source, so they get their own group, `other`, and stay in the training set.
group = extractBefore(string(labelsTable.filename),"_pic");
group(ismissing(group)) = "other";
testGroups = ["996" "5"];
valGroups = ["26" "3_1528505750.77" "15"];
isTest = ismember(group,testGroups);
isVal = ismember(group,valGroups);
isTrain = ~isTest & ~isVal;
splitCounts = table(["Training";"Validation";"Test"],[sum(isTrain);sum(isVal);sum(isTest)], ...
    VariableNames=["Set" "Hands"])
classCounts = array2table([countcats(labels(isTrain))';countcats(labels(isVal))';countcats(labels(isTest))'], ...
    RowNames=["Training" "Validation" "Test"],VariableNames="fingers_" + classNames)
figure
bar(categorical(classNames,classNames),table2array(classCounts)')
legend(["Training" "Validation" "Test"]), xlabel("Number of fingers"), ylabel("Hands")
title("Hands per class and set")
%[text] Every set contains all six classes. The `other` group is a shortcut to watch for: all of it is "5 fingers", so a network could learn "this photo style means 5". The validation and test sets contain no `other` photos, so a low accuracy for 5 fingers there would show it.
XTrain = X(:,:,:,isTrain); labelsTrain = labels(isTrain);
XVal = X(:,:,:,isVal); labelsVal = labels(isVal);
XTest = X(:,:,:,isTest); labelsTest = labels(isTest);
%%
%[text] ## Save the Prepared Data
save(fullfile(projectFolder,"data","preparedData.mat"), ...
    "XTrain","labelsTrain","XVal","labelsVal","XTest","labelsTest","classNames")
%[text] Next, open **Part2\_TrainNetworks**.
%%
%[text] ## Helper Function
function out = ternary(condition,a,b)
if condition, out = a; else, out = b; end
end

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
