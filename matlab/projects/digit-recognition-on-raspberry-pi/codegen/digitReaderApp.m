function digitReaderApp() %#codegen
%digitReaderApp Read handwritten digits on the Raspberry Pi camera until you stop it.
%   The program runs on the Raspberry Pi. It grabs a frame from the camera,
%   calls digitDetect, and prints the result and the time it took.
%   Stop it with Ctrl+C.

r = raspi;
cam = cameraboard(r,"Resolution","640x480");
k = int32(0);
while true
    k = k + 1;
    frame = snapshot(cam);
    t0 = tic;
    [digit,probabilities] = digitDetect(frame);
    ms = toc(t0)*1000;
    if digit < 0
        fprintf("frame %2d: no digit  (%5.1f ms)\n",k,ms);
    else
        fprintf("frame %2d: digit %d  (confidence %3.0f%%, %5.1f ms)\n",k,digit,100*max(probabilities),ms);
    end
    pause(0.1);                    % about 10 frames per second, and no busy loop
end
end
