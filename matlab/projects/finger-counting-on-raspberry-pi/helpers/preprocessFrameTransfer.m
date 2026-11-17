function x = preprocessFrameTransfer(frame) %#codegen
%preprocessFrameTransfer Turn a 480x640 RGB camera frame into a 96x96x3 network input for the pretrained SqueezeNet.
%   Keeps the centered 480x480 square and averages every 5x5 block of pixels.
%   Unlike the small CNN of the tutorial, SqueezeNet expects values from 0 to 255.

f = single(frame);
square = f(:,81:560,:);                       % centered 480 x 480 square
blocks = reshape(square,5,96,5,96,3);         % split into 5 x 5 blocks
x = reshape(mean(mean(blocks,1),3),96,96,3);
end
