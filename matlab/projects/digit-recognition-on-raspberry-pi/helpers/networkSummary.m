function s = networkSummary(net)
%NETWORKSUMMARY Number of learnables and parameter memory of a network.
%   s = networkSummary(net) returns a struct with fields NumLearnables and
%   MemoryKB, computed with estimateNetworkMetrics. For a quantized network
%   the memory reflects the int8 weights. Only the compressible layers
%   (convolution and fully connected) are counted. Projected layers are
%   unpacked first so that estimateNetworkMetrics can see them.

if ~quantizationDetails(net).IsQuantized
    net = unpackProjectedLayers(net);
end
metrics = estimateNetworkMetrics(net);
s.NumLearnables = sum(metrics.NumberOfLearnables);
s.MemoryKB = sum(metrics.("ParameterMemory (MB)")) * 1024;
end
