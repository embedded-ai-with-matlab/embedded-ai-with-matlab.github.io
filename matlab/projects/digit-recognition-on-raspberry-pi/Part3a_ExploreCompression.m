%[text] # Digit Recognition on a Raspberry Pi 5
%[text] ## Part 3a (optional): Explore How Far the Network Can Be Compressed
%[text] Part 3 applies one compression recipe. This optional script shows how that recipe was chosen: it analyzes the network, looks at how many directions each layer really uses, and measures the accuracy of pruning and projection at increasing strength. It takes a few minutes. Run **Part2\_TrainNetwork** first; running **Part3\_CompressNetwork** and **Part4\_QuantizeNetwork** adds your recipe to the last two figures.
%%
%[text] ## Load the Network and the Data
projectFolder = fileparts(mfilename("fullpath"));
if ~isfolder(fullfile(projectFolder,"helpers"))   % the Live Editor can run a temporary copy of the script
    projectFolder = fileparts(matlab.desktop.editor.getActiveFilename);
end
cd(projectFolder)
addpath(fullfile(projectFolder,"helpers"))
load(fullfile(projectFolder,"data","preparedData.mat"))
load(fullfile(projectFolder,"models","baseline.mat"),"net")
rng(0)
summaryOriginal = networkSummary(net);
accuracyOriginal = testnet(net,XTest,labelsTest,"accuracy");
augmenter = imageDataAugmenter(RandRotation=[-12 12],RandXShear=[-12 12],RandScale=[0.8 1.2],RandXTranslation=[-3 3],RandYTranslation=[-3 3]);
dsTrain = augmentedImageDatastore([28 28 1],XTrain,labelsTrain,DataAugmentation=augmenter);
%%
%[text] ## Look at the Analysis for Compression
%[text] Deep Network Designer shows a report when you open the network with `deepNetworkDesigner(net)` and click **Analyze for Compression**. The helper `exportCompressionAnalysis` shows the same report in a separate window and saves it as an image.
try
    exportCompressionAnalysis(net,fullfile(projectFolder,"images","compression-analysis.png"))
    figure
    imshow(fullfile(projectFolder,"images","compression-analysis.png"))
catch err
    disp("Could not create the compression report: " + err.message)
    disp("Open the network with deepNetworkDesigner(net) and click Analyze for Compression instead.")
end
%[text] The report estimates how much of the parameter memory pruning, projection, and quantization could remove at most. These maximums are *structural* limits: how far each technique could go if the network structure were the only constraint, regardless of accuracy.
%%
%[text] ## Where Are the Parameters and the Computation?
metrics = estimateNetworkMetrics(net)
layerNames = categorical(metrics.LayerName,metrics.LayerName);
figure
tiledlayout(1,2)
nexttile
bar(layerNames,metrics.NumberOfLearnables)
set(gca,TickLabelInterpreter="none")
ylabel("Learnable parameters")
title("Memory: learnables per layer")
nexttile
bar(layerNames,metrics.NumberOfMACs/1e3)
set(gca,TickLabelInterpreter="none")
ylabel("Thousands of MACs per image")
title("Computation: MACs per layer")
exportgraphics(gcf,fullfile(projectFolder,"images","compression-layers.png"),Resolution=150)
%[text] The convolution layers hold most of the weights and almost all of the computation, so these are the layers where compression pays off.
%%
%[text] ## See How Many Directions Each Layer Uses
%[text] Projection relies on the observation that the activations of a layer often vary along far fewer directions than the layer has neurons. `neuronPCA` computes the principal components of each layer's activations over the training data. The faster a curve rises to 100%, the more that layer can be projected without losing information.
mbqTrain = minibatchqueue(arrayDatastore(XTrain,IterationDimension=4), ...
    MiniBatchSize=256,MiniBatchFcn=@(x) cat(4,x{:}),MiniBatchFormat="SSCB");
npca = neuronPCA(net,mbqTrain,VerbosityLevel="off");

figure
hold on
for k = 1:numel(npca.LayerNames)-1   % the output layer has only one output per class
    eigenvalues = sort(npca.OutputEigenvalues{k},"descend");
    explained = 100*cumsum(eigenvalues)/sum(eigenvalues);
    plot(100*(1:numel(explained))/numel(explained),explained,LineWidth=1.5,DisplayName=npca.LayerNames(k))
end
hold off
yline(95,":",HandleVisibility="off")
xlabel("Principal components kept (% of layer outputs)")
ylabel("Explained variance (%)")
title("Explained variance of layer activations")
legend(Location="southeast",Interpreter="none")
grid on
exportgraphics(gcf,fullfile(projectFolder,"images","compression-pca.png"),Resolution=150)
%%
%[text] ## Measure the Accuracy Cost of Compression
%[text] Apply each technique at increasing strength, and measure the test accuracy twice: directly ("one-shot") and after fine-tuning. Taylor pruning removes whole convolution filters, and projection keeps the principal directions found by `neuronPCA`.
optionsSweep = trainingOptions("adam", ...
    MaxEpochs=8, ...
    MiniBatchSize=64, ...
    InitialLearnRate=5e-4, ...
    ValidationData={XVal,labelsVal}, ...
    ValidationFrequency=50, ...
    Metrics="accuracy", ...
    Shuffle="every-epoch", ...
    OutputNetwork="best-validation", ...
    Plots="none", ...
    Verbose=false);
levels = [0.1 0.3 0.5 0.7 0.9];

prunableNet = accumulateTaylorScores(net,XTrain,labelsTrain);
pruningSweep = table('Size',[numel(levels) 4],'VariableTypes',repmat("double",1,4), ...
    'VariableNames',["Reduction" "MemoryKB" "OneShotAccuracy" "FineTunedAccuracy"]);
for i = 1:numel(levels)
    numToPrune = round(levels(i)*prunableNet.NumPrunables);
    netPruned = dlnetwork(updatePrunables(prunableNet,MaxToPrune=numToPrune));
    s = networkSummary(netPruned);
    pruningSweep.Reduction(i) = 1 - s.NumLearnables/summaryOriginal.NumLearnables;
    pruningSweep.MemoryKB(i) = s.MemoryKB;
    pruningSweep.OneShotAccuracy(i) = testnet(netPruned,XTest,labelsTest,"accuracy");
    rng(0)
    netPruned = trainnet(dsTrain,netPruned,"crossentropy",optionsSweep);
    pruningSweep.FineTunedAccuracy(i) = testnet(netPruned,XTest,labelsTest,"accuracy");
end
pruningSweep

projectionSweep = pruningSweep;
for i = 1:numel(levels)
    netProjected = compressNetworkUsingProjection(net,npca,LearnablesReductionGoal=levels(i),VerbosityLevel="off");
    s = networkSummary(netProjected);
    projectionSweep.Reduction(i) = 1 - s.NumLearnables/summaryOriginal.NumLearnables;
    projectionSweep.MemoryKB(i) = s.MemoryKB;
    projectionSweep.OneShotAccuracy(i) = testnet(netProjected,XTest,labelsTest,"accuracy");
    rng(0)
    netProjected = trainnet(dsTrain,netProjected,"crossentropy",optionsSweep);
    projectionSweep.FineTunedAccuracy(i) = testnet(netProjected,XTest,labelsTest,"accuracy");
end
projectionSweep
%%
%[text] ## Plot Accuracy Versus Compression
hasRecipe = isfile(fullfile(projectFolder,"models","compressed.mat"));   % true after you run Part 3
if hasRecipe
    load(fullfile(projectFolder,"models","compressed.mat"),"compressionResults")
    chosenReduction = 1 - compressionResults.Learnables(end)/compressionResults.Learnables(1);
    chosenAccuracy = compressionResults.("Test accuracy (%)")(end);
end
colors = orderedcolors("gem");
figure
hold on
plot(100*[0; pruningSweep.Reduction],[accuracyOriginal; pruningSweep.OneShotAccuracy],"--o",Color=colors(1,:),DisplayName="Pruning, one-shot")
plot(100*[0; pruningSweep.Reduction],[accuracyOriginal; pruningSweep.FineTunedAccuracy],"-o",Color=colors(1,:),LineWidth=1.5,MarkerFaceColor=colors(1,:),DisplayName="Pruning, fine-tuned")
plot(100*[0; projectionSweep.Reduction],[accuracyOriginal; projectionSweep.OneShotAccuracy],"--s",Color=colors(2,:),DisplayName="Projection, one-shot")
plot(100*[0; projectionSweep.Reduction],[accuracyOriginal; projectionSweep.FineTunedAccuracy],"-s",Color=colors(2,:),LineWidth=1.5,MarkerFaceColor=colors(2,:),DisplayName="Projection, fine-tuned")
if hasRecipe
    plot(100*chosenReduction,chosenAccuracy,"p",MarkerSize=16,Color="k",MarkerFaceColor=colors(3,:),DisplayName="Part 3 recipe (prune + project)")
end
hold off
xlabel("Learnable parameters removed (%)")
ylabel("Test accuracy (%)")
title("Accuracy versus compression strength")
ylim("padded")
legend(Location="southwest")
grid on
exportgraphics(gcf,fullfile(projectFolder,"images","compression-sweep.png"),Resolution=150)
%[text] Dashed lines: the accuracy right after compression. Solid lines: after fine-tuning. Fine-tuning recovers most of the accuracy, up to a strength where the network has too little capacity left.
%%
%[text] ## Place Your Recipe in the Design Space
hasPath = hasRecipe && isfile(fullfile(projectFolder,"models","quantizationResults.mat"));   % true after you run Part 4
if hasPath
    load(fullfile(projectFolder,"models","quantizationResults.mat"),"results")
    pathMemory = [compressionResults.("Memory (KB)"); results.("Parameter memory (KB)")(end)];
    pathAccuracy = [compressionResults.("Test accuracy (%)"); results.("Test accuracy (%)")(end)];
end
figure
hold on
plot(pruningSweep.MemoryKB,pruningSweep.FineTunedAccuracy,"-o",Color=colors(1,:),DisplayName="Pruning (float32)")
plot(pruningSweep.MemoryKB/4,pruningSweep.FineTunedAccuracy,":o",Color=colors(1,:),DisplayName="Pruning (int8, estimated)")
plot(projectionSweep.MemoryKB,projectionSweep.FineTunedAccuracy,"-s",Color=colors(2,:),DisplayName="Projection (float32)")
plot(projectionSweep.MemoryKB/4,projectionSweep.FineTunedAccuracy,":s",Color=colors(2,:),DisplayName="Projection (int8, estimated)")
if hasPath
    plot(pathMemory,pathAccuracy,"-p",Color="k",LineWidth=1.5,MarkerSize=12,MarkerFaceColor=colors(3,:),DisplayName="Path taken in Parts 3 and 4")
end
hold off
set(gca,XScale="log",XDir="reverse",XTick=[1 2 5 10 20 50 100 200 500])
xlim([max(0.5,min(pruningSweep.MemoryKB)/4*0.7) 1.4*max(pathMemory)])
ylim([min(55,floor(min([pruningSweep.FineTunedAccuracy; projectionSweep.FineTunedAccuracy])-3)) 103])
xlabel("Network parameter memory (KB, log scale)")
ylabel("Test accuracy (%)")
title("Compression design space")
legend(Location="southwest")
grid on
exportgraphics(gcf,fullfile(projectFolder,"images","compression-design-space.png"),Resolution=150)
%[text] Moving to the right means a smaller network. The estimated int8 curves assume that quantization divides the parameter memory by four.
%%
%[text] ## Save the Results
save(fullfile(projectFolder,"models","compressionSweep.mat"),"pruningSweep","projectionSweep")
%[text] Next, open **Part3\_CompressNetwork**.

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
