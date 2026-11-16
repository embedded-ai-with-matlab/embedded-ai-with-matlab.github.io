%[text] # Digit Recognition on a Raspberry Pi 5
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
%[text] Code generation starts from one function, `digitDetect`. It takes a 480 × 640 camera frame, prepares it with `preprocessFrame` (see below), runs the quantized network, and returns the digit and the ten probabilities. If nothing is written in the middle of the frame, it returns −1: "no digit".
type digitDetect
%[text] The network is loaded with `coder.loadDeepLearningNetwork` from the MAT file saved in Part 4. The input is labeled `"SSCB"` (spatial, spatial, channel, batch): code generation for quantized pooling layers requires the batch dimension, even for a single image.
%%
%[text] ## Prepare the Camera Frame
%[text] The camera sees dark ink on white paper, but the network was trained on white digits on a black background, and on digits that are centered. `preprocessFrame` bridges the difference: it keeps the centered 448 × 448 square of the frame, averages 16 × 16 blocks of pixels to get 28 × 28, and estimates the paper with a morphological closing (a maximum filter followed by a minimum filter), which removes the thin dark strokes of the ink but keeps a smooth change of the light, so a shadow is not mistaken for ink. The paper becomes 0 and the ink becomes values up to 1. Finally, it moves the center of mass of the ink to the center of the image. It uses only basic array operations, so MATLAB and the Pi prepare the images in exactly the same way.
type preprocessFrame
%%
%[text] ## Make Test Frames
%[text] The test images are 28 × 28 digits, but the function expects a camera frame. To test the complete function on all 1,500 test digits, each digit is enlarged 16 times, drawn as dark ink on white paper, and placed in the center of a 480 × 640 frame, like a digit on a sheet of paper.
frame = makeFrame(XTest(:,:,1,1));
size(frame)
imshow(frame)
%%
%[text] ## Check the Function in MATLAB
numTest = size(XTest,4);
digitMATLAB = zeros(numTest,1,"int32");
probabilitiesMATLAB = zeros(10,numTest,"single");
for k = 1:numTest
    [digitMATLAB(k),probabilitiesMATLAB(:,k)] = digitDetect(makeFrame(XTest(:,:,1,k)));
end
accuracyFunction = 100*mean(categorical(digitMATLAB,0:9,classNames) == labelsTest)
%[text] The accuracy of the whole function is close to the accuracy of the network, so the preparation of the frame keeps the digit intact. A blank sheet gives "no digit":
digitBlank = digitDetect(uint8(255*ones(480,640,3)))
%%
%[text] ## Build a MEX Function and Compare
%[text] Before generating code for the Pi, build a MEX function with the same deep learning settings. It is generated C++ code that MATLAB runs, so you can compare it with MATLAB on all test frames.
mexCfg = coder.config("mex");
mexCfg.DeepLearningConfig = coder.DeepLearningConfig(TargetLibrary="none");
mexCfg.LargeConstantGeneration = "KeepInSourceFiles";
codegen -config mexCfg digitDetect -args {zeros(480,640,3,"uint8")} -o digitDetect_mex
digitMEX = zeros(numTest,1,"int32");
probabilitiesMEX = zeros(10,numTest,"single");
for k = 1:numTest
    [digitMEX(k),probabilitiesMEX(:,k)] = digitDetect_mex(makeFrame(XTest(:,:,1,k)));
end
digitAgreement = mean(digitMEX == digitMATLAB)
maxDifference = max(abs(probabilitiesMEX(:) - probabilitiesMATLAB(:)))
%[text] The digits must match exactly, and the probabilities may differ only by rounding error.
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
codegen -config cfg digitDetect -args {zeros(480,640,3,"uint8")} -d codegen/lib -report
generated = dir(fullfile("codegen","lib","*.cpp"));
sizeKB = sum([generated.bytes])/1024
%[text] The header file shows the interface that the Pi program calls:
type(fullfile("codegen","lib","digitDetect.h"))
%%
%[text] ## Save the Test Results
save(fullfile(projectFolder,"models","verification.mat"),"digitMATLAB","probabilitiesMATLAB","accuracyFunction")
%[text] Next, open **Part6\_DeployToRaspberryPi**.
%%
%[text] ## Helper Function
function frame = makeFrame(digit)
% Draw a 28 x 28 digit as dark ink, enlarged 16 times, on white paper of a 480 x 640 frame
paper = 255*ones(480,640);
paper(17:464,97:544) = 255*(1 - 0.9*repelem(digit,16,16));
frame = uint8(repmat(paper,1,1,3));
end

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
