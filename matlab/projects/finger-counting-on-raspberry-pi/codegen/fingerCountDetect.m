function [count,probabilities] = fingerCountDetect(frame) %#codegen
%fingerCountDetect Count the fingers in one 480x640 RGB camera frame.
%   The function prepares the frame, runs the quantized network, and returns
%   the number of fingers (0 to 5) and the six class probabilities. It is
%   the entry point for code generation.

persistent net
if isempty(net)
    net = coder.loadDeepLearningNetwork("models/netQ.mat");
end
x = preprocessFrame(frame);                                  % 96 x 96 x 3, values 0 to 1
probabilities = extractdata(predict(net,dlarray(x,"SSCB")));
[~,k] = max(probabilities);
count = int32(k - 1);
end
