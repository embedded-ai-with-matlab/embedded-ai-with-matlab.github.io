%[text] # Finger Counting on a Raspberry Pi 5
%[text] ## Part 6: Deploy to the Raspberry Pi and Run on the Camera
%[text] This script builds the finger counter as a stand-alone program for the Raspberry Pi, runs it on the board with the Pi camera, and checks it against MATLAB. The program grabs a frame from the camera, runs the generated code from Part 5 on it, and prints the number of fingers and the time it took. Nothing is sent to your computer for processing. Run **Part5\_GenerateCode** first.
%%
%[text] ## Set Up the Project
projectFolder = fileparts(mfilename("fullpath"));
if ~isfolder(fullfile(projectFolder,"helpers"))   % the Live Editor can run a temporary copy of the script
    projectFolder = fileparts(matlab.desktop.editor.getActiveFilename);
end
cd(projectFolder)
addpath(fullfile(projectFolder,"helpers"),fullfile(projectFolder,"codegen"))
rng(0)
%%
%[text] ## Connect to the Raspberry Pi
%[text] This needs the **Raspberry Pi Blockset** (Add-On Explorer), and a Raspberry Pi on the same network with the Raspberry Pi Camera attached. Set the connection details of your board. The first connection installs a small MATLAB server on the Pi, and may ask you to run `raspisetup` to install packages that are missing.
%[text] Define `piAddress`, `piUser`, and `piPassword` in the Command Window before you run this section, so that your password is never saved in the script. The values below are used only if the variables do not exist yet.
if ~exist("piAddress","var"), piAddress = "raspberrypi.local"; end    % IP address or host name of your Raspberry Pi
if ~exist("piUser","var"), piUser = "pi"; end                         % Linux user name on the Pi
if ~exist("piPassword","var"), piPassword = ""; end                   % set it in the Command Window; do not save it in the file
if ~exist("r","var") || ~isa(r,"raspi")           % connect only once; clear r to reconnect
    r = raspi(piAddress,piUser,piPassword);
end
r
%%
%[text] ## Check the Camera View
%[text] The network sees the **centered square** of the camera frame. Take one picture and mark that square. Hold your hand inside it, with the fingers spread out and clearly visible, the way the hands in the training photos were.
cam = cameraboard(r,"Resolution","640x480");
frame = snapshot(cam);
imshow(frame)
rectangle(Position=[81 1 480 480],EdgeColor="g",LineWidth=3)
title("The network sees the green square")
%[text] Run this section again after moving your hand or the camera until your hand fills most of the green square. The camera is used by only one program at a time, so release it before the Pi program starts:
clear cam
%%
%[text] ## Build the Program on the Raspberry Pi
%[text] `fingerCounterApp` is the program for the Pi. It creates the camera, then grabs frame after frame and calls `fingerCountDetect` on each, until it is stopped.
type fingerCounterApp
%[text] Configure code generation for a stand-alone program on a 64-bit Raspberry Pi. MATLAB generates the C++ code on your computer, copies it to the Pi, and builds it there with the compiler on the Pi.
hw = coder.hardware("Raspberry Pi (64bit)");
hw.DeviceAddress = r.DeviceAddress;
hw.Username = piUser;
hw.Password = piPassword;
hw.BuildDir = "/home/" + piUser + "/MATLAB_ws/fingercounter";
cfg = coder.config("exe");
cfg.Hardware = hw;
cfg.GenerateExampleMain = "GenerateCodeAndCompile";
cfg.DeepLearningConfig = coder.DeepLearningConfig(TargetLibrary="none");
cfg.LargeConstantGeneration = "KeepInSourceFiles";
cfg.EnableOpenMP = false;
codegen -config cfg fingerCounterApp -d codegen/pi -report
%[text] The board type must match the operating system on the Pi: `Raspberry Pi (64bit)` for a 64-bit system, and `Raspberry Pi` for a 32-bit system.
%%
%[text] ## Run the Program on the Pi
%[text] Start the program on the board for 20 seconds. **Hold up your hand in front of the camera** and change the number of fingers every few seconds, to see how the program follows you. The `timeout` command stops the program after 20 seconds, and `stdbuf -oL` makes it send each line as soon as it is printed. (MATLAB will warn during the build that the program does not end: that is intended.) The output returns to MATLAB when the program ends.
program = strtrim(string(system(r,"find /home/" + piUser + "/MATLAB_ws/fingercounter -name fingerCounterApp.elf | head -1")));
output = system(r,"cd '" + fileparts(program) + "' && stdbuf -oL timeout 20 ./fingerCounterApp.elf 2>/dev/null; true");   % "; true" because timeout ends with an error code
disp(output)
%[text] **To use it live, without a time limit,** open a terminal on your computer, log in to the Pi, and start the program there. Each line appears while the program runs, and Ctrl+C stops it:
%[text] `ssh <user>@<address>` followed by `cd <program folder>` and `./fingerCounterApp.elf 2>/dev/null`
disp("Program folder on the Pi: " + fileparts(program))
%%
%[text] ## Plot the Result
%[text] Read the numbers from the output and plot what the network reported and how long each frame took.
tokens = regexp(output,"frame\s+(\d+):\s+(\d) fingers\s+\(confidence\s+(\d+)%,\s+([\d.]+) ms\)","tokens");
result = str2double(vertcat(tokens{:}));
tiledlayout(3,1,TileSpacing="compact")
nexttile
stairs(result(:,1),result(:,2),LineWidth=2)
ylabel("Fingers"); ylim([-0.5 5.5]); yticks(0:5)
nexttile
plot(result(:,1),result(:,3),LineWidth=2)
ylabel("Confidence (%)"); ylim([0 100])
nexttile
plot(result(:,1),result(:,4),LineWidth=2)
ylabel("Time per frame (ms)"); xlabel("Frame")
meanTimeMs = mean(result(2:end,4))
%[text] The time per frame is only the network and the preparation of the frame; the camera needs additional time for each picture.
%%
%[text] ## Check the Pi Against MATLAB
%[text] Take a picture with MATLAB, and run the MEX function from Part 5 on it. The result should agree with what the Pi program reports for the same scene.
cam = cameraboard(r,"Resolution","640x480");
frame = snapshot(cam);
clear cam
[countMATLAB,probabilitiesMATLAB] = fingerCountDetect_mex(frame)
%%
%[text] ## How Much Does the Program Take?
%[text] Check the size of the program, and the memory it uses while it runs. The memory includes the camera libraries, not only the network.
programKB = str2double(system(r,"stat -L -c %s '" + program + "'"))/1024
peakMemory = system(r,"cd '" + fileparts(program) + "' && (./fingerCounterApp.elf >/dev/null 2>&1 & pid=$!; sleep 2; grep VmHWM /proc/$pid/status; wait $pid)")
%%
%[text] ## What to Expect
%[text] - **Accuracy:** about 60–70% on the held-out groups of the dataset (Part 4). On a new camera, a new room, and your own hand, it can be lower or higher. This is the real test of the model: try 0 to 5 fingers and count how often it is right.
%[text] - **No "no hand" class:** the network always answers with a number from 0 to 5, even if there is no hand in the picture, and it can be very confident about it. Point the camera at the ceiling and see.
%[text] - **Hand placement:** the network was trained on tight crops around a hand. It works best when the hand fills most of the green square. \
%[text] To improve it, collect your own photos with `captureFingerSession` (in the `helpers` folder) in several sessions, add them to the training data in Part 1, and train again.

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
