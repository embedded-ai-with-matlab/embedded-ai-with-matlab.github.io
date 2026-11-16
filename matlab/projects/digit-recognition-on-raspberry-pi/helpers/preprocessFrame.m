function x = preprocessFrame(frame) %#codegen
%preprocessFrame Turn a 480x640 RGB camera frame into a 28x28 digit image.
%   The camera sees dark ink on white paper. The training images show white
%   digits on a black background, so this function
%     1. converts the frame to gray and keeps the centered 448x448 square,
%     2. averages every 16x16 block of pixels to get 28x28 pixels,
%     3. estimates the paper with a morphological closing (a maximum filter
%        followed by a minimum filter over 7x7 pixels), which removes the thin
%        dark strokes of the ink but keeps a smooth change of the light across
%        the paper. The paper becomes 0 and the ink becomes values up to 1,
%        and a shadow or shading does not look like ink,
%     4. moves the center of mass of the ink to the center of the image.
%   If there is hardly any ink in the square, the result is all zeros.
%   It uses only basic array operations, so the same function runs in
%   MATLAB and in the code generated for the Raspberry Pi.

f = single(frame);
gray = (0.299*f(:,:,1) + 0.587*f(:,:,2) + 0.114*f(:,:,3))/255;
square = gray(17:464,97:544);                              % centered 448 x 448 square
small = reshape(mean(mean(reshape(square,16,28,16,28),1),3),28,28);

paper = small;                                             % estimate of the paper: closing of the image
for dy = -3:3                                              % 1. brightest pixel within 3 pixels (removes the dark ink)
    rows = min(max((1:28) + dy,1),28);
    for dx = -3:3
        cols = min(max((1:28) + dx,1),28);
        paper = max(paper,small(rows,cols));
    end
end
bright = paper;
for dy = -3:3                                              % 2. darkest pixel within 3 pixels (restores the shading)
    rows = min(max((1:28) + dy,1),28);
    for dx = -3:3
        cols = min(max((1:28) + dx,1),28);
        paper = min(paper,bright(rows,cols));
    end
end
ink = (paper - small)./max(paper,single(0.1));             % 0 on the paper, larger on the ink
peak = max(ink(:));
if peak < 0.25                                             % nothing written in the square
    x = zeros(28,28,"single");
    return
end
ink = min(ink/peak,1);
ink(ink < 0.1) = 0;                                        % remove paper texture and noise

total = sum(ink(:));
idx = 1:28;
cx = sum(sum(ink,1).*idx)/total;                           % center of mass of the ink
cy = sum(sum(ink,2)'.*idx)/total;
x = circshift(ink,[round(14.5 - cy) round(14.5 - cx)]);
end
