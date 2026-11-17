function fig = plotImageNetworkArchitecture(net,inputSize,titleText)
%plotImageNetworkArchitecture Draw a layer diagram of an image classification network.
%   fig = plotImageNetworkArchitecture(net,inputSize,titleText) draws each
%   layer of the dlnetwork net as a box colored by layer type, with a height
%   that grows with the number of channels. The output size of every layer,
%   computed by passing a zero image of size inputSize ([height width
%   channels]) through the network, is printed below the box.

layers = net.Layers;
X = dlarray(zeros([inputSize 1],"single"),"SSCB");
names = string({layers(2:end).Name});
outputs = cell(1,numel(names));
[outputs{:}] = predict(net,X,Outputs=names);
sizes = [{shapeText(X)} cellfun(@shapeText,outputs,UniformOutput=false)];
channels = [size(X,3) cellfun(@(y) size(y,finddim(y,"C")),outputs)];

style = dictionary( ...
    ["ImageInputLayer" "Convolution2DLayer" "BatchNormalizationLayer" "ReLULayer" "MaxPooling2DLayer" ...
     "GlobalAveragePooling2DLayer" "DropoutLayer" "FullyConnectedLayer" "SoftmaxLayer"], ...
    {{"Input",[0.55 0.55 0.55],0.7} {"2-D convolution",[0.00 0.45 0.74],1.0} ...
     {"Batch normalization",[0.93 0.69 0.13],0.7} {"ReLU",[0.47 0.67 0.19],0.7} {"Max pooling",[0.85 0.33 0.10],0.7} ...
     {"Global average pooling",[0.64 0.08 0.18],0.7} {"Dropout",[0.75 0.75 0.75],0.7} ...
     {"Fully connected",[0.49 0.18 0.56],1.0} {"Softmax",[0.30 0.75 0.93],0.7}});

fig = figure(Theme="light",Color="w",Position=[100 100 1400 430]);
ax = axes(fig,Position=[0.01 0.02 0.98 0.80]);
hold(ax,"on")
axis(ax,"off")
heightOf = @(c) 0.6 + 2.4*sqrt(c/max(channels));
gap = 0.18;
x = 0;
used = strings(0);
legendHandles = gobjects(0);
for k = 1:numel(layers)
    className = split(string(class(layers(k))),".");
    s = style{className(end)};
    [label,color,width] = s{:};
    h = heightOf(channels(k));
    rectangle(ax,Position=[x -h/2 width h],FaceColor=color,EdgeColor="k",LineWidth=0.5);
    if ~ismember(label,used)
        used(end+1) = label; %#ok<AGROW>
        legendHandles(end+1) = patch(ax,NaN,NaN,color,DisplayName=label); %#ok<AGROW>
    end
    detail = layerDetail(layers(k));
    if detail ~= ""
        text(ax,x + width/2,h/2 + 0.12,detail,HorizontalAlignment="center", ...
            VerticalAlignment="bottom",FontSize=8,FontWeight="bold")
    end
    text(ax,x + width/2,-h/2 - 0.12,sizes{k},HorizontalAlignment="center", ...
        VerticalAlignment="top",FontSize=8,Color=[0.3 0.3 0.3])
    if k < numel(layers)
        plot(ax,[x + width, x + width + gap],[0 0],"k",LineWidth=0.75)
    end
    x = x + width + gap;
end
maxHeight = heightOf(max(channels));
axis(ax,[-0.2 x -maxHeight/2-0.5 maxHeight/2+0.75])
legend(ax,legendHandles,Location="southoutside",Orientation="horizontal",Box="off",FontSize=9)
title(ax,titleText,FontSize=12)
hold(ax,"off")
fig.Position(3:4) = [max(round(85*x),900) 340];
end

function txt = shapeText(y)
% Output size as "height x width x channels", or the number of channels when there is no image left.
c = size(y,finddim(y,"C"));
s = finddim(y,"S");
if numel(s) == 2 && prod(size(y,s)) > 1
    txt = size(y,s(1)) + "×" + size(y,s(2)) + "×" + c;
else
    txt = string(c);
end
end

function txt = layerDetail(layer)
% Short description printed above layers that have learnable parameters.
switch class(layer)
    case "nnet.cnn.layer.Convolution2DLayer"
        txt = layer.NumFilters + " filters" + newline + layer.FilterSize(1) + "×" + layer.FilterSize(2);
    case "nnet.cnn.layer.FullyConnectedLayer"
        txt = layer.OutputSize + " outputs";
    case "nnet.cnn.layer.DropoutLayer"
        txt = string(layer.Probability);
    otherwise
        txt = "";
end
end
