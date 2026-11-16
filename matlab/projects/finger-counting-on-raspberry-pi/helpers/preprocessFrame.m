function x = preprocessFrame(frame) %#codegen
%preprocessFrame Turn a 480x640 RGB camera frame into a 96x96x3 network input.
%   The function keeps the centered 480x480 square of the frame and averages
%   every 5x5 block of pixels, so the result has values between 0 and 1.
%   It uses only basic array operations, so the same function runs in
%   MATLAB and in the code generated for the Raspberry Pi.

f = single(frame)/255;
square = f(:,81:560,:);                       % centered 480 x 480 square
blocks = reshape(square,5,96,5,96,3);         % split into 5 x 5 blocks
x = reshape(mean(mean(blocks,1),3),96,96,3);
end
