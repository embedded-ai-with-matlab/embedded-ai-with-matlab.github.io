%[text] # Vibration Severity Monitor on Arduino Nano 33 BLE Sense
%[text] ## Part 1: Prepare Your Own IMU Recordings
%[text] The first tutorial trained on a public ceiling-fan dataset, and on the board the detector reported every window as unknown, because the board and its surroundings looked nothing like the training data. This second tutorial closes that gap: it trains on data recorded **with the same Arduino Nano 33 BLE Sense that runs the model**, so the training data and the deployed sensor match exactly.
%[text] The application is a **vibration severity monitor**. Every second, the board reports how strongly it is vibrating, in three levels, similar to the severity zones used in machine-condition monitoring standards:
%[text] - **Level 0: no vibration**: the board rests undisturbed (recorded as `NORMAL`).
%[text] - **Level 1: light vibration**: short, impulsive bursts (recorded by `TAPPING` the board).
%[text] - **Level 2: strong vibration**: sustained, large motion (recorded by `GENTLE_SHAKING` the board). \
%[text] As in the first tutorial, the detector is a hybrid of a 1-D CNN classifier and an autoencoder, compressed with pruning, projection, and quantization, and deployed as generated C code.
%[text] - **Part 1 — Prepare data** (this script)
%[text] - **Part 2 — Train networks**
%[text] - **Part 3 — Compress the classifier**
%[text] - **Part 4 — Quantize both networks**
%[text] - **Part 5 — Generate code**
%[text] - **Part 6 — Deploy to the Arduino** \
%%
%[text] ## Set Up the Project
projectFolder = fileparts(mfilename("fullpath"));
cd(projectFolder)
addpath(fullfile(projectFolder,"helpers"),fullfile(projectFolder,"codegen"))
rng(0)
%%
%[text] ## About the Recordings
%[text] The data was logged from the board's LSM9DS1 IMU with the Arduino\_LSM9DS1 library, at a nominal 100 Hz. There are 15 recordings of 30 seconds for each class, stored as CSV files in one folder per class. Each row holds a microsecond timestamp, the acceleration in g (`ax`, `ay`, `az`), the angular rate in degrees per second (`gx`, `gy`, `gz`), and the label.
dataFolder = fullfile(projectFolder,"data","raw","Vibration_Dataset");
head(readtable(fullfile(dataFolder,"NORMAL","NORMAL_001.csv")),4)
%%
%[text] ## Check the Sampling
%[text] A real logger is never perfectly regular. Plot the time between consecutive samples for one recording.
T = readtable(fullfile(dataFolder,"TAPPING","TAPPING_001.csv"));
dtMs = diff(double(T.timestamp_us))/1000;
histogram(dtMs,BinWidth=0.5)
xlabel("Time between samples (ms)")
ylabel("Count")
title("Sampling intervals, TAPPING\_001")
%[text] Most intervals are 10 ms, but about 80 per recording are 20 ms: the logger occasionally missed a sample. The network needs evenly spaced samples, so `loadRecordings` uses the timestamps to interpolate each recording onto a uniform 100 Hz grid.
%%
%[text] ## Load and Resample the Recordings
recordings = loadRecordings(dataFolder,100);
summary = groupsummary(recordings,"Class",["mean" "sum"],["Duration" "MissedSamples"]);
summary(:,["Class" "GroupCount" "mean_Duration" "sum_MissedSamples"])
%%
%[text] ## Look at the Signals
%[text] Plot five seconds of acceleration and angular rate from one recording of each class. The mean is removed so that gravity and the gyroscope's bias do not hide the motion.
classNames = categories(recordings.Class);
t = (0:499)/100;
tiledlayout(3,2,TileSpacing="compact")
for k = 1:3
    x = recordings.Signal{find(recordings.Class == classNames{k},1)}(1:500,:);
    x = x - mean(x);
    nexttile
    plot(t,x(:,1:3))
    ylim([-0.6 0.6])
    ylabel("Accel (g)")
    title(replace(classNames{k},"_"," ") + ", acceleration")
    nexttile
    plot(t,x(:,4:6))
    ylim([-60 60])
    ylabel("Gyro (°/s)")
    title(replace(classNames{k},"_"," ") + ", angular rate")
end
xlabel("Time (s)")
%[text] Tapping produces short spikes; shaking produces large, continuous motion on both sensors.
%%
%[text] ## Check the Board Orientation
%[text] The mean acceleration of a recording is the gravity vector, which shows how the board was oriented. The `NORMAL` recordings were made face up and face down, while the shaking recordings cover many angles.
g = recordings.Gravity;
scatter3(g(:,1),g(:,2),g(:,3),40,recordings.Class,"filled")
xlabel("g_x"); ylabel("g_y"); zlabel("g_z")
title("Board orientation per recording")
axis equal
colorbar(Ticks=1:3,TickLabels=replace(classNames,"_"," "))
%[text] If the network saw the raw signal, it could learn the shortcut "tilted board means shaking". Every feature in this tutorial removes the mean of each window first, so the orientation cannot be used.
%%
%[text] ## Split the Data by Recording
%[text] Use 9 recordings per class for training, 3 for validation, and 3 for testing. Splitting whole recordings keeps windows from the same 30 s session, which share orientation and handling, out of different sets.
recordingSplit = splitRecordings(recordings.Class,[9 3 3]);
groupcounts(table(recordingSplit,recordings.Class,VariableNames=["Split" "Class"]),["Split" "Class"])
%%
%[text] ## Cut the Recordings into Windows
%[text] The detector works on one-second windows of 100 samples by 6 channels. Training windows overlap by 50% to give the networks more examples; validation and test windows do not overlap.
windowLength = 100;
isTrain = recordingSplit == "train";
isVal = recordingSplit == "val";
isTest = recordingSplit == "test";
[windowsTrain,idx] = makeWindows(recordings.Signal(isTrain),windowLength,50);
labelsTrain = recordings.Class(isTrain);  labelsTrain = labelsTrain(idx);
[windowsVal,idx] = makeWindows(recordings.Signal(isVal),windowLength,windowLength);
labelsVal = recordings.Class(isVal);      labelsVal = labelsVal(idx);
[windowsTest,idx] = makeWindows(recordings.Signal(isTest),windowLength,windowLength);
labelsTest = recordings.Class(isTest);    labelsTest = labelsTest(idx);
numWindows = [numel(windowsTrain) numel(windowsVal) numel(windowsTest)]
%%
%[text] ## Vibration Energy per Class
%[text] Compute the RMS acceleration of every training window. The three classes are ordered by energy, but the tapping and shaking distributions touch, and the *peak* acceleration of a tap is often higher than that of a shake. This is where a learned model is needed.
rmsAccel = cellfun(@(x) rms(reshape(x(:,1:3) - mean(x(:,1:3)),[],1)),windowsTrain);
peakAccel = cellfun(@(x) max(vecnorm(x(:,1:3) - mean(x(:,1:3)),2,2)),windowsTrain);
gscatter(rmsAccel,peakAccel,labelsTrain)
set(gca,XScale="log",YScale="log")
xlabel("RMS acceleration (g)")
ylabel("Peak acceleration (g)")
title("Vibration energy and peak per window")
%%
%[text] ## Compute Normalization Statistics
%[text] The classifier reads the mean-removed raw window, scaled per channel so that acceleration (about 0.1 g) and angular rate (about 15 °/s) have similar ranges. The autoencoder reads 24 motion indicators per window, computed by `computeMotionIndicators`, and is normalized with statistics from the `NORMAL` training windows only.
centered = cellfun(@(x) x - mean(x,1),windowsTrain,UniformOutput=false);
channelScale = single(std(cat(1,centered{:}),0,1));
indicatorsTrain = cell2mat(cellfun(@computeMotionIndicators,windowsTrain,UniformOutput=false));
isNormalTrain = labelsTrain == "NORMAL";
ciMu = single(mean(indicatorsTrain(isNormalTrain,:)));
ciSigma = single(std(indicatorsTrain(isNormalTrain,:)) + 1e-3);
channelScale
%%
%[text] ## Save the Prepared Data
save(fullfile(projectFolder,"data","preparedData.mat"), ...
    "recordings","recordingSplit","windowsTrain","labelsTrain","windowsVal","labelsVal", ...
    "windowsTest","labelsTest","channelScale","ciMu","ciSigma")
%[text] Next, open **Part2\_TrainNetworks**.

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
