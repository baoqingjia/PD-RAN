
function phase2pdata()
%% PHASE2PDATA - Apply phase correction to Bruker NMR data
%
% This function processes Bruker NMR data by applying phase correction values
% stored in text files. It reads phase correction values (PHC0 and PHC1) from
% text files in the output directory and applies them to the corresponding
% NMR data files in the input directory.
%
% Input directories:
%   - SampleDataDir: Directory containing the original Bruker NMR data
%   - PredictedPhaseDir: Directory containing text files with phase correction values
%
% Text file format:
%   Each text file should be named as 'sampleName_sampleNumber.txt' and contain
%   two values: PHC0 (zero-order phase correction) and PHC1 (first-order phase correction)
%
% Example:
%   phase2pdata()

% Specify the directories for input data and phase correction values
SampleDataDir='D:\MySync\Matlab Data\PD-RAN\Data\samples\topspin_formats_data';
PredictedPhaseDir = 'D:\MySync\Matlab Data\PD-RAN\results\vivo\predicted_phase';

% Read all text files in the phase correction directory
phaseFiles = dir(fullfile(PredictedPhaseDir, '*.txt'));

% Process each phase file
for i = 1:length(phaseFiles)
    % Get the filename without extension
    [~, fileName, ~] = fileparts(phaseFiles(i).name);
    
    % Parse sample name and number
    parts = strsplit(fileName, '_');
    sampleName = parts{1};
    sampleNumber = parts{2};
    
    % Construct sample path
    samplePath = fullfile(SampleDataDir, sampleName, sampleNumber); 
    % Check if sample path exists
    if ~exist(samplePath, 'dir')
        fprintf('Warning: Sample path %s does not exist, skipping processing\n', samplePath);
        continue;
    end
    
    % Read phase correction values
    phaseData = load(fullfile(PredictedPhaseDir, phaseFiles(i).name));
    PHC0 = phaseData(1);
    PHC1 = phaseData(2);
    
    fprintf('Processing sample %s_%s: PHC0=%f, PHC1=%f\n', sampleName, sampleNumber, PHC0, PHC1);

    % Apply phase correction to FID data
    LoadBrukerFID_addphase(samplePath, PHC0, PHC1);

    % Update phase values in parameter files
    procNode = fullfile(samplePath, 'pdata', '1');
    

    % Update proc file
    samplePath1 = [samplePath,'/fid'];
    DECIM = ReadTopspinParam(samplePath1, 'DECIM');
    DSPFVS = ReadTopspinParam(samplePath1, 'DSPFVS');
    DIGMOD = ReadTopspinParam(samplePath1, 'DIGMOD');
    GRPDLY = ReadTopspinParam(samplePath1, 'GRPDLY');
    NrPointsToShift = DetermineBrukerDigitalFilter(DECIM, DSPFVS, DIGMOD,GRPDLY);
    ShiftNum =  round( NrPointsToShift );
    ShiftResidual= NrPointsToShift-ShiftNum;

    UpdateProcFile(fullfile(procNode, 'proc'), -PHC0, -PHC1-ShiftResidual*360);   
    % Update procs file
    UpdateProcFile(fullfile(procNode, 'procs'), -PHC0, -PHC1);
end

end

function [phc0, phc1] = loadPhase(phasePath, sampleName)
%% LOADPHASE - Load phase correction values from a text file
%
% This function loads phase correction values from a text file.
%
% Inputs:
%   phasePath - Path to the directory containing phase files
%   sampleName - Name of the sample
%
% Outputs:
%   phc0 - Zero-order phase correction value
%   phc1 - First-order phase correction value

phase = load(fullfile(phasePath, [sampleName '.txt']));

phc0 = phase(1);
phc1 = phase(2);
end

function UpdateProcFile(filePath, PHC0, PHC1)
%% UPDATEPROCFILE - Update phase correction values in proc/procs files
%
% This function updates the PHC0 and PHC1 parameters in Bruker proc/procs files.
%
% Inputs:
%   filePath - Full path to the proc or procs file
%   PHC0 - Zero-order phase correction value
%   PHC1 - First-order phase correction value

% Check if file exists
if ~exist(filePath, 'file')
    fprintf('Warning: File %s does not exist, skipping processing\n', filePath);
    return;
end

% Read file content
fid = fopen(filePath, 'r');
if fid == -1
    fprintf('Error: Cannot open file %s\n', filePath);
    return;
end

lines = {};
lineCount = 0;
while ~feof(fid)
    lineCount = lineCount + 1;
    lines{lineCount} = fgets(fid);
end
fclose(fid);

% Find and update PHC0 and PHC1 lines
phc0Line = 0;
phc1Line = 0;
for i = 1:lineCount
    if contains(lines{i}, '##$PHC0=')
        phc0Line = i;
    elseif contains(lines{i}, '##$PHC1=')
        phc1Line = i;
    end
end

% If PHC0 and PHC1 lines are found, update them
if phc0Line > 0
    lines{phc0Line} = sprintf('##$PHC0= %g\n', PHC0);
end

if phc1Line > 0
    lines{phc1Line} = sprintf('##$PHC1= %g\n', PHC1);
end

% Write back to file
fid = fopen(filePath, 'w');
if fid == -1
    fprintf('Error: Cannot write to file %s\n', filePath);
    return;
end

for i = 1:lineCount
    fprintf(fid, '%s', lines{i});
end
fclose(fid);

fprintf('Updated file %s: PHC0=%f, PHC1=%f\n', filePath, PHC0, PHC1);
end