%[text] # Part 3a: Explore the Compression Design Space
%[text] Before you commit to a compression recipe in Part 3, it helps to see what each technique *could* do to this network. This optional script estimates, without changing the trained classifier:
%[text] - which layers hold the learnable parameters and the computation,
%[text] - how much each layer could be projected, from the principal component analysis of its activations,
%[text] - how accuracy falls as pruning or projection removes more parameters, with and without fine-tuning, and
%[text] - where every option lands in memory and accuracy, including the int8 quantization of Part 4. \
%[text] It starts with the same report that Deep Network Designer shows when you click **Analyze for Compression**. Run **Part2\_TrainNetworks** first. The script takes a few minutes, most of it spent on short fine-tuning runs.
%%
%[text] ## Load the Data and the Trained Classifier
projectFolder = fileparts(mfilename("fullpath"));
cd(projectFolder)
addpath(fullfile(projectFolder,"helpers"),fullfile(projectFolder,"codegen"))
load(fullfile(projectFolder,"data","preparedData.mat"))
load(fullfile(projectFolder,"models","baseline.mat"),"netCls")
rng(0)

prepare = @(w) cellfun(@(x) (x - mean(x,1))./channelScale,w,UniformOutput=false);
XTrain = prepare(windowsTrain);  TTrain = labelsTrain;
XVal   = prepare(windowsVal);    TVal   = labelsVal;
XTest  = prepare(windowsTest);   TTest  = labelsTest;

summaryOriginal = networkSummary(netCls);
accuracyOriginal = testnet(netCls,XTest,TTest,"accuracy",InputDataFormats="TCB");
%%
%[text] ## Analyze the Network for Compression
%[text] In Deep Network Designer, open the network with `deepNetworkDesigner(netCls)` and click **Analyze for Compression** on the **Designer** tab. The helper `exportCompressionAnalysis` shows the same report in a separate window and saves it as an image. The report estimates the maximum reduction each technique could reach on its own, the smallest network each would produce, and which layers support pruning, projection, and quantization.
try
    exportCompressionAnalysis(netCls,fullfile(projectFolder,"images","compression-analysis.png"))
    imshow(fullfile(projectFolder,"images","compression-analysis.png"))
catch err
    disp("Could not create the compression report: " + err.message)
    disp("Open the network with deepNetworkDesigner(netCls) and click Analyze for Compression instead.")
end
%[text] These maximums are structural limits: how far each technique could go if the network structure were the only constraint, regardless of accuracy. The rest of this script measures what each technique actually costs in accuracy.
%%
%[text] ## Where Are the Parameters and the Computation?
%[text] `estimateNetworkMetrics` lists the compressible layers. Layers with many learnables are the ones worth compressing for memory; layers with many multiply-accumulate operations (MACs) dominate the inference time.
metrics = estimateNetworkMetrics(netCls)
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
ylabel("Thousands of MACs per window")
title("Computation: MACs per layer")
%[text] The second and third convolution layers hold most of the weights and almost all of the computation, so these are the layers where compression pays off.
%%
%[text] ## How Many Directions Does Each Layer Really Use?
%[text] Projection relies on the observation that the activations of a layer often vary along far fewer directions than the layer has neurons. `neuronPCA` computes the principal components of each layer's activations over the training data. The faster a curve below reaches 100%, the more that layer can be projected without losing information.
mbqTrain = minibatchqueue(arrayDatastore(XTrain,OutputType="same"), ...
    MiniBatchSize=256,MiniBatchFcn=@(x) cat(3,x{:}),MiniBatchFormat="TCB");
npca = neuronPCA(netCls,mbqTrain,VerbosityLevel="off");

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
%[text] The fully connected layer `fc_1` is extremely redundant: two of its 32 directions carry more than 99% of the variance. The deepest convolution layer, `conv1d_3`, reaches 95% with a quarter of its directions, and the earlier layers need between 40% and 55%. A projected layer keeps only these directions on both its input and its output side, so keeping half of the directions removes about three quarters of the layer's weights. Compared with the ceiling-fan classifier in the first tutorial, the curves rise faster: separating no, light, and strong vibration is an easier task, and the network has more capacity than it needs.
%%
%[text] ## Sweep the Compression Strength
%[text] Next, apply each technique at increasing strength and measure the test accuracy, once directly ("one-shot") and once after a short fine-tuning run of 15 epochs.
%[text] - For **Taylor pruning**, compute the importance score of every filter once over the training data with the helper `accumulateTaylorScores`, then remove the 10%, 20%, ..., 90% least important filters with `updatePrunables`.
%[text] - For **projection**, reuse the `neuronPCA` result and request learnables reductions from 10% to 90% with `compressNetworkUsingProjection`. \
optionsSweep = trainingOptions("adam", ...
    MaxEpochs=15, ...
    MiniBatchSize=64, ...
    InitialLearnRate=5e-4, ...
    InputDataFormats="TCB", ...
    ValidationData={XVal,TVal}, ...
    Metrics="accuracy", ...
    Shuffle="every-epoch", ...
    OutputNetwork="best-validation", ...
    Plots="none", ...
    Verbose=false);

prunableNet = accumulateTaylorScores(netCls,XTrain,TTrain);
pruneFractions = 0.1:0.1:0.9;
pruningSweep = table('Size',[numel(pruneFractions) 4],'VariableTypes',repmat("double",1,4), ...
    'VariableNames',["Reduction" "MemoryKB" "OneShotAccuracy" "FineTunedAccuracy"]);
for i = 1:numel(pruneFractions)
    numToPrune = round(pruneFractions(i)*prunableNet.NumPrunables);
    netPruned = dlnetwork(updatePrunables(prunableNet,MaxToPrune=numToPrune));
    s = networkSummary(netPruned);
    pruningSweep.Reduction(i) = 1 - s.NumLearnables/summaryOriginal.NumLearnables;
    pruningSweep.MemoryKB(i) = s.MemoryKB;
    pruningSweep.OneShotAccuracy(i) = testnet(netPruned,XTest,TTest,"accuracy",InputDataFormats="TCB");
    rng(0)
    netPruned = trainnet(XTrain,TTrain,netPruned,"crossentropy",optionsSweep);
    pruningSweep.FineTunedAccuracy(i) = testnet(netPruned,XTest,TTest,"accuracy",InputDataFormats="TCB");
end
pruningSweep
%%
projectionGoals = 0.1:0.1:0.9;
projectionSweep = pruningSweep;
for i = 1:numel(projectionGoals)
    netProjected = compressNetworkUsingProjection(netCls,npca, ...
        LearnablesReductionGoal=projectionGoals(i),VerbosityLevel="off");
    s = networkSummary(netProjected);
    projectionSweep.Reduction(i) = 1 - s.NumLearnables/summaryOriginal.NumLearnables;
    projectionSweep.MemoryKB(i) = s.MemoryKB;
    projectionSweep.OneShotAccuracy(i) = testnet(netProjected,XTest,TTest,"accuracy",InputDataFormats="TCB");
    rng(0)
    netProjected = trainnet(XTrain,TTrain,netProjected,"crossentropy",optionsSweep);
    projectionSweep.FineTunedAccuracy(i) = testnet(netProjected,XTest,TTest,"accuracy",InputDataFormats="TCB");
end
projectionSweep
%%
%[text] Plot both sweeps together with the recipe used in Part 3 (50% pruning, then 50% projection, each with fine-tuning). The recipe is marked once you have run Part 3.
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
%[text] Without fine-tuning, both techniques keep 100% accuracy up to about 40% reduction and then drop. With 15 epochs of fine-tuning, pruning keeps 100% test accuracy all the way to 87% reduction, and projection keeps it to about 73% before it collapses. Projection breaks down earlier because it approximates every layer at once, while pruning removes whole filters but keeps the remaining ones intact.
%[text] The Part 3 recipe removes 76% of the learnables, which is a conservative choice for this network: pruning alone could remove more. Keep in mind that the test set contains only three recordings per class, so 100% test accuracy does not guarantee the same result on new recordings. Leaving some capacity in reserve makes the model more robust to how the board is held or mounted.
%%
%[text] ## Map the Design Space: Memory versus Accuracy
%[text] Quantizing the weights to int8 in Part 4 divides the parameter memory by about four. Plotting every option at float32 and at the estimated int8 size shows the full range of choices, and, once you have run Parts 3 and 4, the path actually taken through them.
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
    text(pathMemory,pathAccuracy + 0.02*diff(ylim),["Original" "Pruned" "+ projected" "+ int8"], ...
        VerticalAlignment="bottom",HorizontalAlignment="center",FontWeight="bold",FontSize=9)
end
hold off
set(gca,XScale="log",XDir="reverse",XTick=[0.5 1 2 5 10 20 50])
xlim([0.4 40])
xlabel("Classifier parameter memory (KB, log scale)")
ylabel("Test accuracy (%)")
title("Compression design space")
ylim("padded")
legend(Location="southwest")
grid on
%[text] Moving to the right means a smaller model. Every option here fits easily in the 1 MB flash of the Arduino Nano 33 BLE Sense, so the choice is about accuracy, inference time, and leaving room for the application. Pruning and projection with fine-tuning followed by int8 quantization cut the classifier memory by more than fourteen times and keep 100% test accuracy. The pruning curve shows that an int8 classifier of about 1 KB would also be possible.
%[text] Next, open **Part3\_CompressClassifier** to apply the chosen recipe.

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
