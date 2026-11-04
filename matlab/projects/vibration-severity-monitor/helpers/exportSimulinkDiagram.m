function exportSimulinkDiagram(net,fileName,blocksPerRow)
%exportSimulinkDiagram Save a picture of a network as Simulink layer blocks.
%   exportSimulinkDiagram(net,fileName,blocksPerRow) exports the dlnetwork
%   net to a temporary Simulink model with exportNetworkToSimulink, arranges
%   the layer blocks in rows of blocksPerRow blocks, and prints the network
%   subsystem to the PNG file fileName. The model is closed without saving.

modelName = "networkDiagram";
modelFolder = tempname;
mkdir(modelFolder)
cleanup = onCleanup(@() cleanUp(modelName,modelFolder));
% Frame-based input passes a whole window per time step, which the pooling
% layers of a sequence network need.
% The model cannot update from inside a function because its blocks refer
% to the network variable by name. The diagram does not need an update.
isSequence = isa(net.Layers(1),"nnet.cnn.layer.SequenceInputLayer");
warningState = warning("off","nnet_layerblocks:NetworkToSimulinkExporter:UpdateFailed");
restoreWarning = onCleanup(@() warning(warningState));
exportNetworkToSimulink(net,ModelName=modelName,ModelPath=modelFolder,FrameBased=isSequence);
subsystem = modelName + "/" + modelName;

% Order the blocks by following the connections from the input port.
blocks = string(find_system(subsystem,SearchDepth=1,Type="Block"));
blocks(blocks == subsystem) = [];
ordered = blocks(get_param(blocks,"BlockType") == "Inport");
while true
    ports = get_param(ordered(end),"PortConnectivity");
    next = [ports.DstBlock];
    if isempty(next)
        break
    end
    ordered(end+1) = string(getfullname(next(1))); %#ok<AGROW>
end

% Lay the blocks out in rows that read left to right.
delete_line(find_system(subsystem,SearchDepth=1,FindAll="on",Type="line"))
blockWidth = 140;  blockHeight = 50;  columnStep = 200;  rowStep = 120;
for k = 1:numel(ordered)
    column = mod(k-1,blocksPerRow);
    row = floor((k-1)/blocksPerRow);
    if get_param(ordered(k),"BlockType") == "Inport" || get_param(ordered(k),"BlockType") == "Outport"
        width = 30;  height = 16;
    else
        width = blockWidth;  height = blockHeight;
    end
    x = column*columnStep + (blockWidth - width)/2;
    y = row*rowStep + (blockHeight - height)/2;
    set_param(ordered(k),Position=[x y x+width y+height])
end
for k = 1:numel(ordered)-1
    add_line(subsystem,get_param(ordered(k),"Name") + "/1",get_param(ordered(k+1),"Name") + "/1", ...
        Autorouting="smart")
end
print("-s" + subsystem,"-dpng","-r150",fileName)
end

function cleanUp(modelName,modelFolder)
if bdIsLoaded(modelName)
    close_system(modelName,0)
end
rmdir(modelFolder,"s")
end
