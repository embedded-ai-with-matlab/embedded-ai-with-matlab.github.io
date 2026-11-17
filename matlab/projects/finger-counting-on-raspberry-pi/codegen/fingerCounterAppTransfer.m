function fingerCounterAppTransfer() %#codegen
%fingerCounterAppTransfer Count fingers on the Raspberry Pi camera with the transfer-learning network, until stopped.
%   The program runs on the Raspberry Pi. It grabs a frame from the camera,
%   calls fingerCountDetectTransfer, and prints the result and the time it
%   took. Stop it with Ctrl+C.

r = raspi;
cam = cameraboard(r,"Resolution","640x480");
k = int32(0);
while true
    k = k + 1;
    frame = snapshot(cam);
    t0 = tic;
    [count,probabilities] = fingerCountDetectTransfer(frame);
    ms = toc(t0)*1000;
    fprintf("frame %2d: %d fingers  (confidence %3.0f%%, %5.1f ms)\n",k,count,100*max(probabilities),ms);
    pause(0.1);                    % about 10 frames per second, and no busy loop
end
end
