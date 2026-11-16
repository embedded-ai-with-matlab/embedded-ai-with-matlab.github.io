%[text] # Digit Recognition on a Raspberry Pi 5
%[text] ## Part 4: Quantize the Network
%[text] **Quantization** stores the weights and the activations of the network as 8-bit integers instead of 32-bit floating-point numbers. This cuts the parameter memory to a quarter, and integer arithmetic is cheaper than floating-point arithmetic. A **calibration** step runs training images through the network to find the range of values in every layer. Run **Part3\_CompressNetwork** first.
%%
%[text] ## Load the Compressed Network
projectFolder = fileparts(mfilename("fullpath"));
if ~isfolder(fullfile(projectFolder,"helpers"))   % the Live Editor can run a temporary copy of the script
    projectFolder = fileparts(matlab.desktop.editor.getActiveFilename);
end
cd(projectFolder)
addpath(fullfile(projectFolder,"helpers"))
load(fullfile(projectFolder,"data","preparedData.mat"))
load(fullfile(projectFolder,"models","baseline.mat"),"net")
load(fullfile(projectFolder,"models","compressed.mat"),"netCompressed")
rng(0)
%%
%[text] ## Calibrate and Quantize
%[text] A `dlquantizer` object with `ExecutionEnvironment="MATLAB"` targets generic C and C++ code, which is what runs on the Pi in Part 6. Calibrate with every third training image.
quantObj = dlquantizer(netCompressed,ExecutionEnvironment="MATLAB");
prepareNetwork(quantObj);
calibrate(quantObj,augmentedImageDatastore([28 28 1],XTrain(:,:,:,1:3:end)));
netQ = quantize(quantObj,ExponentScheme="histogram");
quantizationDetails(netQ)
%%
%[text] ## Compare the Networks
accuracyOriginal = testnet(net,XTest,labelsTest,"accuracy");
accuracyCompressed = testnet(netCompressed,XTest,labelsTest,"accuracy");
accuracyQ = testnet(netQ,XTest,labelsTest,"accuracy");
summaries = [networkSummary(net); networkSummary(netCompressed); networkSummary(netQ)];
stageNames = ["Original (float32)" "Pruned + projected (float32)" "Pruned + projected + quantized (int8)"];
stage = categorical(stageNames,stageNames)';
memoryKB = [summaries.MemoryKB]';
results = table(stage,[summaries.NumLearnables]',memoryKB,[accuracyOriginal; accuracyCompressed; accuracyQ], ...
    VariableNames=["Stage" "Learnables" "Parameter memory (KB)" "Test accuracy (%)"])
figure
bar(stage,memoryKB)
ylabel("Parameter memory (KB)")
title(sprintf("Network memory reduced by %.0f%%",100*(1 - memoryKB(end)/memoryKB(1))))
%[text] Check that quantization changed the predictions of the compressed network only a little:
predictedCompressed = onehotdecode(minibatchpredict(netCompressed,XTest),classNames,2);
predictedQ = onehotdecode(double(minibatchpredict(netQ,XTest)),classNames,2);   % the quantized network returns fixed-point values
agreement = 100*mean(predictedCompressed == predictedQ)
%%
%[text] ## Save the Quantized Network
save(fullfile(projectFolder,"models","netQ.mat"),"netQ")
save(fullfile(projectFolder,"models","quantizationResults.mat"),"results")
%[text] Next, open **Part5\_GenerateCode**.

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
