function captureFingerSession(cam,sessionName,options)
%captureFingerSession Record one session of finger-count images from the Pi camera.
%   captureFingerSession(cam,"session01") asks you to hold up 0, 1, ..., 5
%   fingers in front of the camera and saves ImagesPerClass images for each,
%   in data/raw/fingers/<sessionName>/<count>/ next to the project.
%
%   A session is one sitting. To get a useful test, record each session
%   with something different: the light, the background, the distance, the
%   hand you use, or the person. The tutorial splits the data by session.
%
%   cam is a cameraboard object, created with cameraboard(raspi(...)).

arguments
    cam
    sessionName (1,1) string
    options.ImagesPerClass (1,1) double = 25
    options.CountdownSeconds (1,1) double = 5
    options.SecondsBetweenImages (1,1) double = 0.6
    options.Counts (1,:) double = 0:5
end

projectFolder = fileparts(fileparts(mfilename("fullpath")));
sessionFolder = fullfile(projectFolder,"data","raw","fingers",sessionName);
assert(~isfolder(sessionFolder),"%s already exists. Use a new session name.",sessionFolder)

for n = options.Counts
    classFolder = fullfile(sessionFolder,string(n));
    mkdir(classFolder)
    fprintf("\n=== Hold up %d finger(s). Starting in %d s. Move your hand a little between images. ===\n", ...
        n,options.CountdownSeconds)
    for s = options.CountdownSeconds:-1:1
        fprintf("%d... ",s); pause(1)
    end
    fprintf("\n")
    for k = 1:options.ImagesPerClass
        frame = snapshot(cam);
        imwrite(frame,fullfile(classFolder,sprintf("%s_%d_%03d.jpg",sessionName,n,k)),Quality=95)
        pause(options.SecondsBetweenImages)
    end
    fprintf("Saved %d images for %d finger(s).\n",options.ImagesPerClass,n)
end
fprintf("\nSession %s done: %s\n",sessionName,sessionFolder)
end
