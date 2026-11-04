function fig = plotNetworkArchitecture(net,inputSize,titleText)
%plotNetworkArchitecture Draw a layer diagram of a feed-forward network.
%   fig = plotNetworkArchitecture(net,inputSize,titleText) draws each layer
%   of the dlnetwork net as a box colored by layer type, with a height that
%   grows with the number of channels. The output size of every layer,
%   computed by passing a zero input of size inputSize through the network,
%   is printed below the box. For a sequence network, inputSize is
%   [numTimeSteps numChannels]; for a feature network, it is numFeatures.

layers = net.Layers;
isSequence = isa(layers(1),"nnet.cnn.layer.SequenceInputLayer");
if isSequence
    X = dlarray(zeros(inputSize(1),inputSize(2),1,"single"),"TCB");
else
    X = dlarray(zeros(inputSize,1,"single"),"CB");
end
names = string({layers(2:end).Name});
outputs = cell(1,numel(names));
[outputs{:}] = predict(net,X,Outputs=names);
sizes = [{shapeText(X)} cellfun(@shapeText,outputs,UniformOutput=false)];
channels = [size(X,finddim(X,"C")) cellfun(@(y) size(y,finddim(y,"C")),outputs)];

style = dictionary( ...
    ["SequenceInputLayer" "FeatureInputLayer" "Convolution1DLayer" "BatchNormalizationLayer" ...
     "ReLULayer" "MaxPooling1DLayer" "GlobalAveragePooling1DLayer" "FullyConnectedLayer" "SoftmaxLayer"], ...
    {{"Input",[0.55 0.55 0.55],0.6} {"Input",[0.55 0.55 0.55],0.6} {"1-D convolution",[0.00 0.45 0.74],1.0} ...
     {"Batch normalization",[0.93 0.69 0.13],0.3} {"ReLU",[0.47 0.67 0.19],0.3} {"Max pooling",[0.85 0.33 0.10],0.45} ...
     {"Global average pooling",[0.64 0.08 0.18],0.6} {"Fully connected",[0.49 0.18 0.56],1.0} {"Softmax",[0.30 0.75 0.93],0.6}});

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
% Output size as "time x channels" or "features".
c = size(y,finddim(y,"C"));
t = finddim(y,"T");
if isempty(t) || size(y,t) == 1 && ~contains(dims(y),"T")
    txt = string(c);
else
    txt = size(y,t) + "×" + c;
end
end

function txt = layerDetail(layer)
% Short description printed above layers that have learnable parameters.
switch class(layer)
    case "nnet.cnn.layer.Convolution1DLayer"
        txt = layer.NumFilters + " filters" + newline + "size " + layer.FilterSize;
    case "nnet.cnn.layer.FullyConnectedLayer"
        txt = layer.OutputSize + " outputs";
    otherwise
        txt = "";
end
end
