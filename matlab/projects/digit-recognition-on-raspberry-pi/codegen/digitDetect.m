function [digit,probabilities] = digitDetect(frame) %#codegen
%digitDetect Read the handwritten digit in one 480x640 RGB camera frame.
%   The function prepares the frame, runs the quantized network, and returns
%   the digit (0 to 9) and the ten class probabilities. If nothing is
%   written in the center of the frame, it returns -1 and zero probabilities.
%   It is the entry point for code generation.

persistent net
if isempty(net)
    net = coder.loadDeepLearningNetwork("models/netQ.mat");
end
x = preprocessFrame(frame);                                  % 28 x 28, ink = 1, paper = 0
if ~any(x(:))                                                % blank paper
    digit = int32(-1);
    probabilities = zeros(10,1,"single");
    return
end
probabilities = extractdata(predict(net,dlarray(x,"SSCB")));
[~,k] = max(probabilities);
digit = int32(k - 1);
end
