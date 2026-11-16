%[text] # Digit Recognition on a Raspberry Pi 5
%[text] ## Part 6: Deploy to the Raspberry Pi and Read Digits with the Camera
%[text] This script tests the digit reader on your own handwriting, builds it as a stand-alone program for the Raspberry Pi, and runs it on the board with the Pi camera. The program grabs a frame from the camera, runs the generated code from Part 5 on it, and prints the digit and the time it took. Nothing is sent to your computer for processing. Run **Part5\_GenerateCode** first.
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
%[text] This needs the **Raspberry Pi Blockset** (Add-On Explorer), and a Raspberry Pi on the same network with the Raspberry Pi Camera attached. The first connection installs a small MATLAB server on the Pi, and may ask you to run `raspisetup` to install packages that are missing.
%[text] Define `piAddress`, `piUser`, and `piPassword` in the Command Window before you run this section, so that your password is never saved in the script. The values below are used only if the variables do not exist yet.
if ~exist("piAddress","var"), piAddress = "raspberrypi.local"; end    % IP address or host name of your Raspberry Pi
if ~exist("piUser","var"), piUser = "pi"; end                         % Linux user name on the Pi
if ~exist("piPassword","var"), piPassword = ""; end                   % set it in the Command Window; do not save it in the file
if ~exist("r","var") || ~isa(r,"raspi")           % connect only once; clear r to reconnect
    r = raspi(piAddress,piUser,piPassword);
end
r
%%
%[text] ## Prepare the Paper and the Camera View
%[text] Write a digit in the middle of a white sheet of paper with a **thick black marker or a thick dark pen**, as large as the green square allows. The network sees only the centered square of the camera frame, shown as a green box. Thin pencil strokes, a small digit, or shadows across the paper make the result worse. Light the paper evenly, and hold it still, facing the camera.
cam = cameraboard(r,"Resolution","640x480");
frame = snapshot(cam);
imshow(frame)
rectangle(Position=[97 17 448 448],EdgeColor="g",LineWidth=3)
title("The network reads the green square")
%[text] Run this section again after moving the paper or the camera until the digit fills most of the green square. Then look at what the network sees, after the preparation of the frame:
x = preprocessFrame(frame);
imshow(x)
title("What the network sees")
%%
%[text] ## Measure Your Accuracy
%[text] The test set of Part 2 contains digits written like the training digits. Your handwriting, your pen, and the camera are different. This section measures how well the network reads *your* digits: for each digit from 0 to 9, it counts you down for 8 seconds, and then takes five pictures. Write the digit on a sheet, and hold it in the green square when the countdown ends. The pictures are read with the MEX function from Part 5.
if ~exist("measureAccuracy","var"), measureAccuracy = true; end    % set to false to skip this section
secondsToSwitch = 8;                                                 % time to put the next digit in the green square
digitsTrue = repelem((0:9)',5);
digitsRead = zeros(50,1);
confidence = zeros(50,1);
k = 0;
digitList = 0:9;
if ~measureAccuracy, digitList = []; end
for d = digitList
    fprintf("\n=== Show the digit %d. Starting in %d s ===\n",d,secondsToSwitch)
    pause(secondsToSwitch)
    for n = 1:5
        k = k + 1;
        [digitsRead(k),p] = digitDetect_mex(snapshot(cam));
        confidence(k) = max(p);
        pause(0.3)
    end
    fprintf("Read: %s\n",num2str(digitsRead(k-4:k)'))
end
if measureAccuracy
    accuracyLive = 100*mean(digitsRead == digitsTrue)
    noDigit = 100*mean(digitsRead < 0)
end
%[text] Plot what the network read for each digit that you showed:
if measureAccuracy
    figure
    confusionchart(categorical(digitsTrue),categorical(digitsRead),Title="Digits read from your camera")
end
clear cam          % the camera can be used by only one program at a time
%%
%[text] ## Build the Program on the Raspberry Pi
%[text] `digitReaderApp` is the program for the Pi. It creates the camera, then grabs frame after frame and calls `digitDetect` on each, until it is stopped.
type digitReaderApp
%[text] Configure code generation for a stand-alone program on a 64-bit Raspberry Pi. MATLAB generates the C++ code on your computer, copies it to the Pi, and builds it there with the compiler on the Pi. The board type must match the operating system on the Pi: `Raspberry Pi (64bit)` for a 64-bit system, and `Raspberry Pi` for a 32-bit system. MATLAB warns that the program does not end: that is intended.
hw = coder.hardware("Raspberry Pi (64bit)");
hw.DeviceAddress = r.DeviceAddress;
hw.Username = piUser;
hw.Password = piPassword;
hw.BuildDir = "/home/" + piUser + "/MATLAB_ws/digitreader";
cfg = coder.config("exe");
cfg.Hardware = hw;
cfg.GenerateExampleMain = "GenerateCodeAndCompile";
cfg.DeepLearningConfig = coder.DeepLearningConfig(TargetLibrary="none");
cfg.LargeConstantGeneration = "KeepInSourceFiles";
cfg.EnableOpenMP = false;
codegen -config cfg digitReaderApp -d codegen/pi -report
%%
%[text] ## Run the Program on the Pi
%[text] Start the program on the board for 20 seconds. **Show a digit to the camera**, and change it every few seconds. The `timeout -s KILL 20` command stops the program after 20 seconds, and `stdbuf -oL` makes it send each line as soon as it is printed. The output returns to MATLAB when the program ends.
program = strtrim(string(system(r,"find /home/" + piUser + "/MATLAB_ws/digitreader -name digitReaderApp.elf | head -1")));
output = system(r,"cd '" + fileparts(program) + "' && stdbuf -oL timeout -s KILL 20 ./digitReaderApp.elf 2>/dev/null; true");   % "; true" because timeout ends with an error code
system(r,"pkill -9 -f '[d]igitReaderApp.elf'; true");   % make sure that no copy of the program keeps running
lines = splitlines(string(output));
disp(strjoin(lines([1:min(10,end) end-min(2,end-1):end]),newline))      % the first and the last lines
%[text] **To use it live, without a time limit,** open a terminal on your computer, log in to the Pi, and start the program there. Each line appears while the program runs, and Ctrl+C stops it:
%[text] `ssh <user>@<address>` followed by `cd <program folder>` and `./digitReaderApp.elf 2>/dev/null`
disp("Program folder on the Pi: " + fileparts(program))
%%
%[text] ## Plot the Result
%[text] Read the numbers from the output and plot what the program reported and how long each frame took.
isDigit = contains(lines,"digit") & ~contains(lines,"no digit");
tokens = regexp(lines(isDigit),"frame\s+(\d+):\s+digit (\d)\s+\(confidence\s+(\d+)%,\s+([\d.]+) ms\)","tokens","once");
result = zeros(0,4);
if ~isempty(tokens), result = str2double(vertcat(tokens{:})); end
timeTokens = regexp(lines,"\(\s*([\d.]+) ms\)","tokens","once");
timesMs = str2double(vertcat(timeTokens{:}));
if ~isempty(result)
    figure
    tiledlayout(2,1,TileSpacing="compact")
    nexttile
    plot(result(:,1),result(:,2),".",MarkerSize=14)
    ylabel("Digit read"); ylim([-0.5 9.5]); yticks(0:9); xlabel("Frame")
    nexttile
    plot(timesMs,LineWidth=2)
    ylabel("Time per frame (ms)"); xlabel("Frame")
end
meanTimeMs = mean(timesMs(2:end))
framesWithDigit = size(result,1)
%[text] The time per frame is only the preparation of the frame and the network; the camera needs additional time for each picture.
%%
%[text] ## How Much Does the Program Take?
%[text] Check the size of the program, and the memory it uses while it runs. The memory includes the camera libraries, not only the network.
programKB = str2double(system(r,"stat -L -c %s '" + program + "'"))/1024
peakMemory = system(r,"cd '" + fileparts(program) + "' && (./digitReaderApp.elf >/dev/null 2>&1 & pid=$!; sleep 2; grep VmHWM /proc/$pid/status; wait $pid)")
%%
%[text] ## What to Expect
%[text] - **Accuracy:** about 98% on the test images of the data set (Part 4). The accuracy on your own handwriting is the number from "Measure Your Accuracy". It is higher with thick dark strokes, large digits, even light, and a paper that is held still.
%[text] - **Blank paper** gives "no digit". Anything else in the square that is darker than the paper, such as a shadow, a fold, or a printed line, can be read as a digit.
%[text] - **Digits that look alike** (1 and 7, 4 and 9, 3 and 5) are the most likely to be confused. \

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
