function [count,probabilities] = fingerCountDetectTransfer(frame) %#codegen
%fingerCountDetectTransfer Count the fingers in one 480x640 RGB camera frame with the transfer-learning network.
%   Entry point for code generation in Part 8. It prepares the frame, runs the
%   pruned SqueezeNet saved in Part 7, and returns the number of fingers
%   (0 to 5) and the six class probabilities.

persistent net
if isempty(net)
    net = coder.loadDeepLearningNetwork("models/netTransfer.mat");
end
x = preprocessFrameTransfer(frame);                          % 96 x 96 x 3, values 0 to 255
probabilities = extractdata(predict(net,dlarray(x,"SSCB")));
[~,k] = max(probabilities);
count = int32(k - 1);
end
