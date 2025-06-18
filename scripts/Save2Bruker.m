function [outputArg1,outputArg2] = Save2Bruker(PhaseDataAfterphc,pDataDir)
%% Save2Bruker - Save phase-corrected data to Bruker format files
%
% This function saves the phase-corrected data to Bruker format files (1r and 1i)
%
% Input parameters:
%   PhaseDataAfterphc - Phase-corrected data
%   pDataDir - Directory path to save the data
fileID = fopen(fullfile(pDataDir,'/1r'),'w');
realfid = real(PhaseDataAfterphc);
% imagfid = imag(ValidFid);
outdata = [];
for i = 1:length(PhaseDataAfterphc)
    outdata= [outdata realfid(i)];
end
fwrite(fileID,outdata,'int32','l');
fclose(fileID);

fileID = fopen(fullfile(pDataDir,'/1i'),'w');
% realfid = real(PhaseDataAfterphc);
imagfid = imag(PhaseDataAfterphc);
outdata = [];
for i = 1:length(PhaseDataAfterphc)
    outdata= [outdata imagfid(i)];
end
fwrite(fileID,outdata,'int32','l');
fclose(fileID);
end

