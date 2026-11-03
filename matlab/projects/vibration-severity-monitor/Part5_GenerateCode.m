%[text] # Part 5: Generate C Code for the Arm Cortex-M4
%[text] This script turns the complete vibration severity detector, including preprocessing, both quantized networks, and the decision logic, into portable C code with MATLAB® Coder™ and Embedded Coder®, and verifies it against MATLAB.
%[text] Run **Part4\_QuantizeNetworks** first.
%%
%[text] ## The Entry-Point Function
%[text] `vibrationSeverityDetect` (in the `codegen` folder) takes one 100-by-6 window of raw IMU data and returns the vibration level (0, 1, or 2), the three class probabilities, the autoencoder score, and a flag for unrecognized vibration.
projectFolder = fileparts(mfilename("fullpath"));
cd(projectFolder)
addpath(fullfile(projectFolder,"helpers"),fullfile(projectFolder,"codegen"))
load(fullfile(projectFolder,"data","preparedData.mat"),"windowsTest","labelsTest")
type vibrationSeverityDetect
%%
%[text] ## Run the Entry Point in MATLAB
numTest = numel(windowsTest);
levelMatlab = zeros(numTest,1,"int32");
scoreMatlab = zeros(numTest,1,"single");
for i = 1:numTest
    [levelMatlab(i),~,scoreMatlab(i)] = vibrationSeverityDetect(windowsTest{i});
end
clear vibrationSeverityDetect   % reset the persistent networks
accuracyEntryPoint = 100*mean(levelMatlab == int32(double(labelsTest) - 1))
%%
%[text] ## Configure Code Generation
%[text] The settings match the first tutorial: plain C with no third-party libraries, network weights embedded in the source (the board has no file system), no dynamic memory, and the Arm Cortex-M device type.
cfg = coder.config("lib","ecoder",true);
cfg.TargetLang = "C";
cfg.GenCodeOnly = true;
cfg.GenerateExampleMain = "DoNotGenerate";
cfg.DeepLearningConfig = coder.DeepLearningConfig(TargetLibrary="none");
cfg.HardwareImplementation.ProdHWDeviceType = "ARM Compatible->ARM Cortex-M";
cfg.LargeConstantGeneration = "KeepInSourceFiles";
cfg.EnableOpenMP = false;
cfg.EnableDynamicMemoryAllocation = false;
%%
%[text] ## Verify the Generated Code with a MEX Function
mexCfg = coder.config("mex");
mexCfg.DeepLearningConfig = coder.DeepLearningConfig(TargetLibrary="none");
codegen -config mexCfg vibrationSeverityDetect -args {zeros(100,6,"single")} -o vibrationSeverityDetect_mex
levelMex = zeros(numTest,1,"int32");
scoreMex = zeros(numTest,1,"single");
for i = 1:numTest
    [levelMex(i),~,scoreMex(i)] = vibrationSeverityDetect_mex(windowsTest{i});
end
levelAgreement = mean(levelMex == levelMatlab)
medianRelativeDifferenceScore = median(abs(scoreMex - scoreMatlab)./scoreMatlab)
load(fullfile(projectFolder,"models","deploymentParams.mat"),"aeThreshold")
flagAgreement = mean((scoreMex > aeThreshold) == (scoreMatlab > aeThreshold))
%%
%[text] ## Generate the C Library for the Arduino
codegen -config cfg vibrationSeverityDetect -args {zeros(100,6,"single")} -d codegen/build -report
generatedHeader = fileread(fullfile("codegen","build","vibrationSeverityDetect.h"));
signature = regexp(generatedHeader,"extern void vibrationSeverityDetect\([^;]*\);","match","once")
%[text] The input window is stored column by column: 100 samples of `ax`, then `ay`, `az`, `gx`, `gy`, and `gz`.
%%
%[text] ## Copy the Generated Code into the Arduino Sketch
sketchFolder = fullfile(projectFolder,"arduino","VibrationSeverityMonitor");
modelFolder = fullfile(sketchFolder,"src","model");
if isfolder(modelFolder)
    rmdir(modelFolder,"s")
end
mkdir(modelFolder)
copyfile(fullfile("codegen","build","*.c"),modelFolder)
copyfile(fullfile("codegen","build","*.h"),modelFolder)
generatedFiles = string({dir(fullfile(modelFolder,"*.c")).name})'
%[text] Next, open **Part6\_DeployToArduino**.

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
