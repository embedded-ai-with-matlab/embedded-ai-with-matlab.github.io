%[text] # Finger Counting on a Raspberry Pi 5
%[text] ## Part 5: Generate C++ Code
%[text] The Raspberry Pi does not run MATLAB. **MATLAB Coder** translates a MATLAB function, together with the quantized network, into C++ code that the Pi compiles and runs on its own. This script prepares that function, checks it against MATLAB, and generates the code. Run **Part4\_QuantizeNetwork** first.
%%
%[text] ## Set Up the Project
projectFolder = fileparts(mfilename("fullpath"));
if ~isfolder(fullfile(projectFolder,"helpers"))   % the Live Editor can run a temporary copy of the script
    projectFolder = fileparts(matlab.desktop.editor.getActiveFilename);
end
cd(projectFolder)
addpath(fullfile(projectFolder,"helpers"),fullfile(projectFolder,"codegen"))
load(fullfile(projectFolder,"data","preparedData.mat"),"XTest","labelsTest","classNames")
load(fullfile(projectFolder,"models","netQ.mat"),"netQ")
rng(0)
%%
%[text] ## The Entry-Point Function
%[text] Code generation starts from one function, `fingerCountDetect`. It takes a 480 × 640 camera frame, prepares it with `preprocessFrame` (the same function that prepared the training images), runs the quantized network, and returns the number of fingers and the six probabilities.
type fingerCountDetect
%[text] The network is loaded with `coder.loadDeepLearningNetwork` from the MAT file saved in Part 4. The input is labeled `"SSCB"` (spatial, spatial, channel, batch): code generation for quantized pooling layers requires the batch dimension, even for a single image.
%%
%[text] ## Make Test Frames
%[text] The test images are 96 × 96 crops, but the function expects a camera frame. To test the complete function on the 247 test hands, each crop is enlarged five times and placed in the center of an empty 480 × 640 frame. Preparing that frame gives back the original crop exactly.
makeFrame = @(crop) cat(2,zeros(480,80,3,"uint8"),repelem(crop,5,5),zeros(480,80,3,"uint8"));
frame = makeFrame(XTest(:,:,:,1));
size(frame)
%%
%[text] ## Check the Function in MATLAB
numTest = size(XTest,4);
countMATLAB = zeros(numTest,1,"int32");
probabilitiesMATLAB = zeros(6,numTest,"single");
for k = 1:numTest
    [countMATLAB(k),probabilitiesMATLAB(:,k)] = fingerCountDetect(makeFrame(XTest(:,:,:,k)));
end
accuracyFunction = 100*mean(categorical(countMATLAB,0:5,classNames) == labelsTest)
%%
%[text] ## Build a MEX Function and Compare
%[text] Before generating code for the Pi, build a MEX function with the same deep learning settings. It is generated C++ code that MATLAB runs, so you can compare it with MATLAB on all test frames.
mexCfg = coder.config("mex");
mexCfg.DeepLearningConfig = coder.DeepLearningConfig(TargetLibrary="none");
mexCfg.LargeConstantGeneration = "KeepInSourceFiles";
codegen -config mexCfg fingerCountDetect -args {zeros(480,640,3,"uint8")} -o fingerCountDetect_mex
%[text] Run the MEX function on all test frames and compare.
countMEX = zeros(numTest,1,"int32");
probabilitiesMEX = zeros(6,numTest,"single");
for k = 1:numTest
    [countMEX(k),probabilitiesMEX(:,k)] = fingerCountDetect_mex(makeFrame(XTest(:,:,:,k)));
end
countAgreement = mean(countMEX == countMATLAB)
medianRelativeDifference = median(abs(probabilitiesMEX(:) - probabilitiesMATLAB(:))./max(abs(probabilitiesMATLAB(:)),1e-6))
%[text] The counts must match exactly, and the probabilities may differ only by rounding error.
%%
%[text] ## Generate C++ Code
%[text] Generate a C++ library for a 64-bit ARM processor, such as the one in the Raspberry Pi 5. `TargetLibrary="none"` produces plain C++ without third-party libraries, and `LargeConstantGeneration="KeepInSourceFiles"` puts the network weights in the source files, so the code does not depend on separate data files.
cfg = coder.config("lib");
cfg.TargetLang = "C++";
cfg.GenCodeOnly = true;
cfg.GenerateExampleMain = "DoNotGenerate";
cfg.DeepLearningConfig = coder.DeepLearningConfig(TargetLibrary="none");
cfg.LargeConstantGeneration = "KeepInSourceFiles";
cfg.HardwareImplementation.ProdHWDeviceType = "ARM Compatible->ARM 64-bit (LP64)";
cfg.EnableOpenMP = false;
codegen -config cfg fingerCountDetect -args {zeros(480,640,3,"uint8")} -d codegen/lib -report
generated = dir(fullfile("codegen","lib","*.cpp"));
sizeKB = sum([generated.bytes])/1024
%[text] The header file shows the interface that the Pi program calls:
type(fullfile("codegen","lib","fingerCountDetect.h"))
%%
%[text] ## Save the Test Results
save(fullfile(projectFolder,"models","verification.mat"),"countMATLAB","probabilitiesMATLAB","accuracyFunction")
%[text] Next, open **Part6\_DeployToRaspberryPi**.

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
