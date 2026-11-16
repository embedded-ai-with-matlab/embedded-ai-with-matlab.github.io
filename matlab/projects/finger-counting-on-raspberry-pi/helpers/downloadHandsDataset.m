function datasetFolder = downloadHandsDataset(rawFolder)
%downloadHandsDataset Download the Hands Dataset from GitHub if it is missing.
%   The dataset (Apache License 2.0, https://github.com/SubhiH/HandsDataset)
%   has about 1,550 photos with labeled hands showing 0 to 5 fingers. The
%   download is about 430 MB. The photos show a person's face, so they are
%   not included in this repository.

datasetFolder = fullfile(rawFolder,"HandsDataset-master");
if isfolder(fullfile(datasetFolder,"images"))
    return
end
if ~isfolder(rawFolder), mkdir(rawFolder), end
zipFile = fullfile(tempdir,"HandsDataset.zip");
fprintf("Downloading the Hands Dataset (about 430 MB)...\n")
websave(zipFile,"https://github.com/SubhiH/HandsDataset/archive/refs/heads/master.zip");
unzip(zipFile,rawFolder)
delete(zipFile)
end
