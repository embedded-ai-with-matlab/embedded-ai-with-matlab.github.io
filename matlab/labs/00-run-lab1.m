% Launch the full first lab from any MATLAB working directory.
repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
projectRoot = fullfile(repoRoot,'matlab','projects','vibration-severity-monitor');
run(fullfile(projectRoot,'Part1_PrepareData.m'));
run(fullfile(repoRoot,'matlab','labs','01-vibration-baseline.m'));
