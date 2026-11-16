function fingerCounterApp() %#codegen
%fingerCounterApp Count fingers on the Raspberry Pi camera until you stop it.
%   The program runs on the Raspberry Pi. It grabs a frame from the camera,
%   calls fingerCountDetect, and prints the result and the time it took.
%   Stop it with Ctrl+C.

r = raspi;
cam = cameraboard(r,"Resolution","640x480");
k = int32(0);
while true
    k = k + 1;
    frame = snapshot(cam);
    t0 = tic;
    [count,probabilities] = fingerCountDetect(frame);
    ms = toc(t0)*1000;
    fprintf("frame %2d: %d fingers  (confidence %3.0f%%, %5.1f ms)\n",k,count,100*max(probabilities),ms);
end
end
