%[text] # Finger Counting on a Raspberry Pi 5
%[text] ## Part 8 (continuation): Deploy the Transfer-Learning Network
%[text] This script generates C++ code for the pruned SqueezeNet of Part 7, checks it against MATLAB, builds it on the Raspberry Pi, and measures how long it takes there. It follows Parts 5 and 6; only the network and the preparation of the frame change. Run **Part7\_TransferLearning** first.
%%
%[text] ## Set Up the Project
projectFolder = fileparts(mfilename("fullpath"));
if ~isfolder(fullfile(projectFolder,"helpers"))   % the Live Editor can run a temporary copy of the script
    projectFolder = fileparts(matlab.desktop.editor.getActiveFilename);
end
cd(projectFolder)
addpath(fullfile(projectFolder,"helpers"),fullfile(projectFolder,"codegen"))
load(fullfile(projectFolder,"data","preparedData.mat"),"XTest","labelsTest","classNames")
rng(0)
%%
%[text] ## The Entry-Point Function
%[text] `fingerCountDetectTransfer` is the counterpart of `fingerCountDetect` from Part 5. It loads the pruned SqueezeNet, and its preparation of the frame, `preprocessFrameTransfer`, keeps the values between 0 and 255, as SqueezeNet expects.
type fingerCountDetectTransfer
%%
%[text] ## Check the Function and Build a MEX Function
%[text] As in Part 5, each 96 × 96 test crop is enlarged five times and placed in the center of an empty 480 × 640 frame, which the preparation turns back into the original crop.
makeFrame = @(crop) cat(2,zeros(480,80,3,"uint8"),repelem(crop,5,5),zeros(480,80,3,"uint8"));
numTest = size(XTest,4);
countMATLAB = zeros(numTest,1,"int32");
probabilitiesMATLAB = zeros(6,numTest,"single");
for k = 1:numTest
    [countMATLAB(k),probabilitiesMATLAB(:,k)] = fingerCountDetectTransfer(makeFrame(XTest(:,:,:,k)));
end
accuracyFunction = 100*mean(categorical(countMATLAB,0:5,classNames) == labelsTest)
%[text] Build a MEX function with the same deep learning settings, and compare it with MATLAB on all test frames.
mexCfg = coder.config("mex");
mexCfg.DeepLearningConfig = coder.DeepLearningConfig(TargetLibrary="none");
mexCfg.LargeConstantGeneration = "KeepInSourceFiles";
codegen -config mexCfg fingerCountDetectTransfer -args {zeros(480,640,3,"uint8")} -o fingerCountDetectTransfer_mex
countMEX = zeros(numTest,1,"int32");
probabilitiesMEX = zeros(6,numTest,"single");
for k = 1:numTest
    [countMEX(k),probabilitiesMEX(:,k)] = fingerCountDetectTransfer_mex(makeFrame(XTest(:,:,:,k)));
end
countAgreement = mean(countMEX == countMATLAB)
maxDifference = max(abs(probabilitiesMEX(:) - probabilitiesMATLAB(:)))
%[text] The counts must match exactly, and the probabilities may differ only by rounding error.
%%
%[text] ## Connect to the Raspberry Pi
%[text] Define `piAddress`, `piUser`, and `piPassword` in the Command Window before you run this section, so that your password is never saved in the script. The values below are used only if the variables do not exist yet.
if ~exist("piAddress","var"), piAddress = "raspberrypi.local"; end    % IP address or host name of your Raspberry Pi
if ~exist("piUser","var"), piUser = "pi"; end                         % Linux user name on the Pi
if ~exist("piPassword","var"), piPassword = ""; end                   % set it in the Command Window; do not save it in the file
if ~exist("r","var") || ~isa(r,"raspi")           % connect only once; clear r to reconnect
    r = raspi(piAddress,piUser,piPassword);
end
r
%%
%[text] ## Build the Program on the Pi
%[text] `fingerCounterAppTransfer` is the counterpart of `fingerCounterApp` from Part 6. The build takes a minute or two.
hw = coder.hardware("Raspberry Pi (64bit)");
hw.DeviceAddress = r.DeviceAddress;
hw.Username = piUser;
hw.Password = piPassword;
hw.BuildDir = "/home/" + piUser + "/MATLAB_ws/fingercountertransfer";
cfg = coder.config("exe");
cfg.Hardware = hw;
cfg.GenerateExampleMain = "GenerateCodeAndCompile";
cfg.DeepLearningConfig = coder.DeepLearningConfig(TargetLibrary="none");
cfg.LargeConstantGeneration = "KeepInSourceFiles";
cfg.EnableOpenMP = false;
codegen -config cfg fingerCounterAppTransfer -d codegen/pi -report
%%
%[text] ## Run the Program on the Pi
%[text] Start the program for 12 seconds and hold up your hand. The `timeout -s KILL 12` command stops the program, and `pkill` makes sure that no copy keeps running.
program = strtrim(string(system(r,"find /home/" + piUser + "/MATLAB_ws/fingercountertransfer -name fingerCounterAppTransfer.elf | head -1")));
output = system(r,"cd '" + fileparts(program) + "' && stdbuf -oL timeout -s KILL 12 ./fingerCounterAppTransfer.elf 2>/dev/null; true");
system(r,"pkill -9 -f '[f]ingerCounterAppTransfer.elf'; true");
lines = splitlines(string(output));
lines = lines(contains(lines,"frame"));
disp(strjoin(lines([1:min(8,end) end-min(2,end-1):end]),newline))      % the first and the last lines
tokens = regexp(lines,"\(confidence\s+(\d+)%,\s+([\d.]+) ms\)","tokens","once");
values = str2double(vertcat(tokens{:}));
meanTimeMs = mean(values(2:end,2))
%[text] The time is for the preparation of the frame and the network. For comparison, the small int8 network of Part 6 needs about 60 ms for the same work on the same board.
%%
%[text] ## How Much Does the Program Take?
programKB = str2double(system(r,"stat -L -c %s '" + program + "'"))/1024
peakMemory = system(r,"cd '" + fileparts(program) + "' && (./fingerCounterAppTransfer.elf >/dev/null 2>&1 & pid=$!; sleep 3; grep VmHWM /proc/$pid/status; kill -9 $pid; wait $pid 2>/dev/null; true)")
system(r,"pkill -9 -f '[f]ingerCounterAppTransfer.elf'; true");
%%
%[text] ## What to Expect
%[text] - **No "no hand" class:** like the small CNN, this network always answers with a number. Cover the lens or point the camera at a plain wall, and it can still report a finger count with a fairly high confidence (in a test: "4 fingers" at about 70% for a plain grey wall). The confidence is not a reliable hand detector. To fix this, add a "no hand" class with photos of empty scenes.
%[text] - **Accuracy on your hand:** the accuracy of about 87% on the test groups is for the data set of Part 1. Your camera, room, and hand can make it better or worse: try 0 to 5 fingers, and count how often it is right.
%[text] - **int8:** the quantization of SqueezeNet works, but running the quantized network in MATLAB needs the Image Processing Toolbox, so this part stays in float32. The program is 1.6 MB, which is small for a Raspberry Pi. \

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
