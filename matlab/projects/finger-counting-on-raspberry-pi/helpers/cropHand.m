function crop = cropHand(img,box,outSize,margin)
%cropHand Cut a square around a labeled hand and resize it.
%   crop = cropHand(img,[xmin ymin xmax ymax],outSize,margin) takes the
%   bounding box of a hand, makes it square (using the longer side) and
%   adds margin (for example 0.2 = 20%) on every side. Where the square
%   leaves the photo, the edge pixels are repeated. The result is an
%   outSize-by-outSize image.

arguments
    img (:,:,3) uint8
    box (1,4) double
    outSize (1,1) double = 96
    margin (1,1) double = 0.2
end
cx = (box(1) + box(3))/2;
cy = (box(2) + box(4))/2;
side = max(box(3) - box(1),box(4) - box(2))*(1 + 2*margin);
cols = round(cx - side/2):round(cx + side/2);
rows = round(cy - side/2):round(cy + side/2);
cols = min(max(cols,1),size(img,2));      % repeat edge pixels outside the photo
rows = min(max(rows,1),size(img,1));
crop = imresize(img(rows,cols,:),[outSize outSize]);
end
