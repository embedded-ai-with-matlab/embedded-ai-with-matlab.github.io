function score = autoencoderScore(net,X)
%AUTOENCODERSCORE Reconstruction error of an autoencoder.
%   score = autoencoderScore(net,X) returns the mean squared reconstruction
%   error for each row of X (observations-by-features). Inputs that look
%   like the normal training data reconstruct well (low score); unfamiliar
%   inputs reconstruct poorly (high score).

Xhat = single(minibatchpredict(net,X));   % quantized networks return fi values
score = mean((single(X) - Xhat).^2,2);
end
