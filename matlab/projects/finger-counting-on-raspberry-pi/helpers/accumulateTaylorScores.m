function prunableNet = accumulateTaylorScores(net,X,T)
%accumulateTaylorScores Compute Taylor importance scores over an image data set.
%   prunableNet = accumulateTaylorScores(net,X,T) converts the dlnetwork
%   net to a TaylorPrunableNetwork and accumulates the first-order Taylor
%   importance score of every prunable filter over the images X (an array of
%   size height-by-width-by-channels-by-numImages) with categorical labels
%   T. Call updatePrunables on the result to remove the least important
%   filters without retraining.

classes = categories(T);
prunableNet = taylorPrunableNetwork(net);
mbq = minibatchqueue(combine(arrayDatastore(X,IterationDimension=4),arrayDatastore(T)), ...
    MiniBatchSize=128, ...
    MiniBatchFcn=@(x,t) deal(cat(4,x{:}),onehotencode(cat(1,t{:}),2,ClassNames=classes)'), ...
    MiniBatchFormat=["SSCB" "CB"]);
while hasdata(mbq)
    [XBatch,TBatch] = next(mbq);
    [activations,gradients] = dlfeval(@taylorGradients,prunableNet,XBatch,TBatch);
    prunableNet = updateScore(prunableNet,activations,gradients);
end
end

function [activations,gradients] = taylorGradients(prunableNet,X,T)
[Y,~,activations] = forward(prunableNet,X);
loss = crossentropy(Y,T);
gradients = dlgradient(loss,activations);
end
