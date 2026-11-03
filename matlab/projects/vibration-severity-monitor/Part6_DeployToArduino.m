%[text] # Part 6: Deploy the Vibration Severity Monitor
%[text] This script builds the Arduino sketch that wraps the generated C code, flashes it to the board, and plots the live output.
%[text] The sketch `arduino/VibrationSeverityMonitor/VibrationSeverityMonitor.ino`:
%[text] - keeps the Arduino\_LSM9DS1 library's default settings (119 Hz output data rate, ±4 g, ±2000 °/s), which are the settings the training data was recorded with,
%[text] - samples acceleration and angular rate every 10 ms and collects one-second windows of 100 samples,
%[text] - calls the generated `vibrationSeverityDetect` function and lights the RGB LED green (level 0), yellow (level 1), red (level 2), or blue (unrecognized vibration),
%[text] - shows blue only after 3 consecutive unrecognized windows, so that a single small bump, such as a knock on the desk or a touch on the USB cable, does not make the LED flicker (change `kUnrecognizedWindows` in the sketch to adjust this),
%[text] - prints one CSV line per window over USB serial. \
%[text] **Requirements:** Arduino IDE 2 (which includes `arduino-cli`), the *Arduino Mbed OS Nano Boards* core, and the *Arduino\_LSM9DS1* library. Run **Part5\_GenerateCode** first. This script is written for macOS; on Windows or Linux, change `arduinoCli` to the location of `arduino-cli` on your computer.
%%
%[text] ## Locate the Tools and the Board
projectFolder = fileparts(mfilename("fullpath"));
cd(projectFolder)
sketchFolder = fullfile(projectFolder,"arduino","VibrationSeverityMonitor");
arduinoCli = "/Applications/Arduino IDE.app/Contents/Resources/app/lib/backend/resources/arduino-cli";
fqbn = "arduino:mbed_nano:nano33ble";
ports = serialportlist("available");
ports = ports(contains(ports,"usbmodem") | startsWith(ports,"COM"));
port = ports(1);
if ismac
    port = replace(port,"/dev/tty.","/dev/cu.");   % use the call-out device on macOS
end
port
%%
%[text] ## Compile and Upload the Sketch
runCli = @(args) system(sprintf('"%s" %s',arduinoCli,args));
runCli("core install arduino:mbed_nano");
runCli("lib install Arduino_LSM9DS1");
[status,buildLog] = runCli(sprintf('compile --fqbn %s "%s"',fqbn,sketchFolder));
assert(status == 0,buildLog)
memoryUsage = splitlines(string(strtrim(buildLog)));
memoryUsage(end-1:end)
[status,uploadLog] = runCli(sprintf('upload --fqbn %s -p %s "%s"',fqbn,port,sketchFolder));
assert(status == 0,uploadLog)
%%
%[text] ## Monitor the Detector Live
%[text] Read 30 windows (about 30 seconds) from the board. **While this section runs, leave the board still for about ten seconds, then tap it for ten seconds, then shake it gently for ten seconds.**
pause(3)    % let the board restart after flashing
board = serialport(port,115200);
configureTerminator(board,"CR/LF")
board.Timeout = 10;
flush(board)
numWindows = 30;
results = table(Size=[numWindows 5], ...
    VariableTypes=["double" "double" "string" "double" "double"], ...
    VariableNames=["TimeMs" "Level" "State" "AEScore" "LatencyUs"]);
k = 0;
while k < numWindows
    fields = split(readline(board),",");
    if numel(fields) ~= 8 || isnan(str2double(fields(1)))
        continue    % skip the header and any partial line
    end
    k = k + 1;
    results(k,:) = {str2double(fields(1)),str2double(fields(2)),fields(3), ...
        str2double(fields(7)),str2double(fields(8))};
end
clear board
head(results)
%%
%[text] ## Plot the Detector Output
load(fullfile(projectFolder,"models","deploymentParams.mat"),"aeThreshold")
t = (results.TimeMs - results.TimeMs(1))/1000;
tiledlayout(2,1,TileSpacing="compact")
nexttile
stairs(t,results.Level,LineWidth=1.5)
ylim([-0.3 2.3])
yticks(0:2)
yticklabels(["0 none" "1 light" "2 strong"])
ylabel("Vibration level")
title("1-D CNN classifier on the board")
nexttile
semilogy(t,results.AEScore,"-o")
yline(aeThreshold,"--","Threshold")
xlabel("Time (s)")
ylabel("Reconstruction error")
title("Autoencoder on the board")
%%
%[text] ## Inference Time on the Microcontroller
medianLatencyMs = median(results.LatencyUs)/1000
%%
%[text] ## Next Steps
%[text] - **Add more levels or events**, for example a fan or motor running under the board, by recording 15 more sessions in the same format and adding a class in Part 2.
%[text] - **Reduce the delay** by running inference on overlapping windows: keep sampling into a second buffer while the detector runs.
%[text] - **Report levels wirelessly** over Bluetooth Low Energy with the ArduinoBLE library. \

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
