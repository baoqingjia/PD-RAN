% LOAD BRUKER FID - Extract spectra and phase data from Bruker NMR files
% This script processes Bruker TopSpin format NMR data and extracts
% complex spectra and phase correction parameters to text files
clc;close all;clear;

% Define input and output directories
maindir ='data/samples/data_ori/topspin_formats_data';
savePath_spectra ='data/samples/data_ori/Extracted_spectra';
savePath_phase ='data/samples/data_ori/Extracted_phase';

% Create output directories if they don't exist
if ~exist(savePath_spectra, 'dir')
    mkdir(savePath_spectra);
end

if ~exist(savePath_phase, 'dir')
    mkdir(savePath_phase);
end

% Iterate through all subdirectories
subdir  = dir(maindir);

for i = 1:length( subdir )
    % Skip hidden directories and files
    if( isequal( subdir( i ).name, '..')||...
            startsWith(subdir( i ).name, '.') ||....
            ~subdir( i ).isdir)
        continue;
    end
    maindir1 = fullfile( maindir, subdir( i ).name);
    subdir1  = dir( maindir1 );
    for k = 1:length( subdir1 )
        % Skip current and parent directories
        if( isequal( subdir1( k ).name, '.' )||...
                isequal( subdir1( k ).name, '..')||...
                startsWith(subdir1( k ).name, '.') ||....
                ~subdir( k ).isdir)
            continue;
        end 
    fiddirpath = fullfile( maindir1, subdir1( k ).name);

%% Load FID data from binary file
    fidPath = [fiddirpath,'/fid'];
    swHz = ReadTopspinParam(fidPath, 'SW_h');        % Spectral width in Hz
    fidpoints = ReadTopspinParam(fidPath, 'TD');     % Time domain points
    SizeTD1 = 1;
    ByteOrder = 2;
    [fid, SizeTD2, SizeTD1] = GetFIdFromBidary(fidPath, fidpoints, SizeTD1, ByteOrder);

    % Read processing parameters
    specPath = [fiddirpath ,'/pdata/1/1r'];
    swppm = ReadTopspinParam(specPath, 'SW');         % Spectral width in ppm
    offsetppm = ReadTopspinParam(specPath, 'OFFSET'); % Offset in ppm
    lockppm = ReadTopspinParam(specPath, 'LOCKPPM');  % Lock frequency
    swppmHalf=swppm/2;
    IdealWater=offsetppm-swppmHalf;
    Diff=-(IdealWater-lockppm);
%% Calculate digital filter correction
    % Read digital filter parameters
    DECIM = ReadTopspinParam(fidPath, 'DECIM');
    DSPFVS = ReadTopspinParam(fidPath, 'DSPFVS');
    DIGMOD = ReadTopspinParam(fidPath, 'DIGMOD');
    GRPDLY = ReadTopspinParam(fidPath, 'GRPDLY');
    % Calculate points to shift for digital filter correction
    NrPointsToShift = DetermineBrukerDigitalFilter(DECIM, DSPFVS, DIGMOD,GRPDLY);
    ShiftNum =  round( NrPointsToShift );
    ShiftResidual= NrPointsToShift-ShiftNum;
%% Apply exponential window function
    lb = ReadTopspinParam(specPath, 'LB');           % Line broadening parameter
    fidSize=length(fid);
    points = 0:1:(fidSize-1);
    t=exp(-points.*(pi*lb/swHz));                    % Exponential decay
    WindowFidData=fid.*t;
%% Zero padding for FFT
    ftSize = ReadTopspinParam(specPath, 'SI');       % Size for FFT
    if(ftSize~=64*1024)
        ftSize=64*1024;                              % Default to 64K points
    end
    fidAddWin = [WindowFidData zeros(1,ftSize-fidSize)];
%% Apply digital filter point shifting
    TempFidData=fidAddWin(:,1:ShiftNum);
    FidData=[fidAddWin(:, ShiftNum+1:end) TempFidData];
%% Perform FFT and phase correction
    DataBeforePhase1 = ifftshift((ifft(FidData(1,:))));
    % Read phase correction parameters
    PHC0_orig = ReadTopspinParam(specPath, 'PHC0');  % Zero-order phase
    PHC1_orig = ReadTopspinParam(specPath, 'PHC1');  % First-order phase
    % Convert to radians and adjust for digital filter
    PHC0_orig = PHC0_orig*(pi/180.0);
    PHC1_orig = PHC1_orig*(pi/180.0)+ShiftResidual*360*(pi/180.0);
    x = ((0:length(DataBeforePhase1)-1))/length(DataBeforePhase1);
    PhaseDataAfterphc = DataBeforePhase1 .* exp(-1i*(PHC0_orig+PHC1_orig*x));
    
    % Store phase correction values in degrees
    gt_ph0_ph1 = zeros(1,2);
    gt_ph0_ph1(:, 1) = PHC0_orig*(180.0/pi);         % PHC0 in degrees
    gt_ph0_ph1(:, 2) = PHC1_orig*(180.0/pi);         % PHC1 in degrees 

    % Generate filename from directory structure
    an = strsplit(fiddirpath, filesep);
    str1 = char(an{end-1});
    str2 = char(an{end});
    A = [str1, '_', str2];

    % Save complex spectrum data
    data_real_row = real(DataBeforePhase1);
    data_imag_row = imag(DataBeforePhase1);
    filename = fullfile(savePath_spectra, sprintf('%s.txt', A));
    fid = fopen(filename, 'w');
    fprintf(fid, '%f%+fi\n', [data_real_row; data_imag_row]);  
    fclose(fid);

    % Save phase correction parameters
    filename = fullfile(savePath_phase, sprintf('%s.txt', A));  
    fid = fopen(filename, 'w');
    fprintf(fid, '%f ', gt_ph0_ph1);
    fprintf(fid, '\n');  
    fclose(fid);  

    disp('over')
    end
end