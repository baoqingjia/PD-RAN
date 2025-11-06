
function phase2pdata()
%% PHASE2PDATA - Apply phase correction to Bruker NMR data
%
% Reads phase correction values (PHC0, PHC1) from text files and applies
% them to corresponding Bruker NMR data files.
%
% Input: Text files named 'sampleName_sampleNumber.txt' containing PHC0 and PHC1 values
% Output: Updated Bruker data files with corrected phase parameters

% Set input and output directories
SampleDataDir='F:\cg\download\PD-RAN-main\data\samples\topspin_formats_data\Metabolomics';
PredictedPhaseDir = 'F:\cg\download\PD-RAN-main\results\vivo\data_ori\predicted_phase';

% Get all phase correction files
phaseFiles = dir(fullfile(PredictedPhaseDir, '*.txt'));

% Process each file
for i = 1:length(phaseFiles)
    % Extract filename
    [~, fileName, ~] = fileparts(phaseFiles(i).name);
    
    % Parse sample info
    parts = strsplit(fileName, '_');
    sampleName = parts{1};
    
    % Build sample path
    samplePath = fullfile(SampleDataDir, sampleName); 
    % Verify path exists
    if ~exist(samplePath, 'dir')
        fprintf('Warning: Sample path %s does not exist, skipping processing\n', samplePath);
        continue;
    end
    
    % Load phase values
    phaseData = load(fullfile(PredictedPhaseDir, phaseFiles(i).name));
    PHC0 = phaseData(1);
    PHC1 = phaseData(2);
    
    fprintf('Processing sample %s_%s: PHC0=%f, PHC1=%f\n', sampleName, PHC0, PHC1);

    % Apply phase correction
    LoadBrukerFID_addphase(samplePath, PHC0, PHC1);

    % Update parameter files
    procNode = fullfile(samplePath, 'pdata', '1');
    
    % Update proc file
    samplePath1 = [samplePath,'/fid'];
    DECIM = ReadTopspinParam(samplePath1, 'DECIM');
    DSPFVS = ReadTopspinParam(samplePath1, 'DSPFVS');
    DIGMOD = ReadTopspinParam(samplePath1, 'DIGMOD');
    GRPDLY = ReadTopspinParam(samplePath1, 'GRPDLY');
    NrPointsToShift = DetermineBrukerDigitalFilter_Mod(DECIM, DSPFVS, DIGMOD,GRPDLY);
    ShiftNum =  round( NrPointsToShift );
    ShiftResidual= NrPointsToShift-ShiftNum;

    UpdateProcFile(fullfile(procNode, 'proc'), PHC0, PHC1+ShiftResidual*360);   
    % Update procs file
    UpdateProcFile(fullfile(procNode, 'procs'), PHC0, PHC1);
end

end

function [phc0, phc1] = loadPhase(phasePath, sampleName)
%% LOADPHASE - Load phase correction values from text file
%
% Inputs: phasePath (directory), sampleName (sample name)
% Outputs: phc0 (zero-order phase), phc1 (first-order phase)

phase = load(fullfile(phasePath, [sampleName '.txt']));

phc0 = phase(1);
phc1 = phase(2);
end

function UpdateProcFile(filePath, PHC0, PHC1)
%% UPDATEPROCFILE - Update phase values in Bruker proc/procs files
%
% Inputs: filePath (file path), PHC0 (zero-order phase), PHC1 (first-order phase)

% Check file exists
if ~exist(filePath, 'file')
    fprintf('Warning: File %s does not exist, skipping processing\n', filePath);
    return;
end

% Read file
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

% Find PHC0 and PHC1 lines
phc0Line = 0;
phc1Line = 0;
for i = 1:lineCount
    if contains(lines{i}, '##$PHC0=')
        phc0Line = i;
    elseif contains(lines{i}, '##$PHC1=')
        phc1Line = i;
    end
end

% Update found lines
if phc0Line > 0
    lines{phc0Line} = sprintf('##$PHC0= %g\n', PHC0);
end

if phc1Line > 0
    lines{phc1Line} = sprintf('##$PHC1= %g\n', PHC1);
end

% Write updated file
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