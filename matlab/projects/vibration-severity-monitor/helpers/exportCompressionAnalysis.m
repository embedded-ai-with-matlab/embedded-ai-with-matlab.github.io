function exportCompressionAnalysis(net,fileName)
%exportCompressionAnalysis Save the Analyze for Compression report as a PNG.
%   exportCompressionAnalysis(net,fileName) shows the report that Deep
%   Network Designer displays when you click Analyze for Compression, for
%   the dlnetwork net, in a separate window and saves it to fileName with
%   exportapp. The report view is an internal component of Deep Learning
%   Toolbox Model Compression Library and can change between releases; if
%   it is not available, open the network with deepNetworkDesigner(net)
%   and click Analyze for Compression instead.

fig = uifigure(Position=[20 20 1600 1250],Theme="light",Name="Analysis for Network Compression");
cleanup = onCleanup(@() delete(fig));
grid = uigridlayout(fig,[1 1]);
view = dlcompression.internal.app.view.CompressibilityPluginView(grid);
view.Network = net;
drawnow
pause(2)   % let the charts finish rendering
drawnow
exportapp(fig,fileName)
end
