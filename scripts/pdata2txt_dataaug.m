%% PDATA2TXT_DATAAUG - NMR Data Augmentation for Training
%
% Processes Bruker NMR data and generates augmented training samples
% by applying random phase corrections to create multiple variants
% of each spectrum for deep learning model training.
%
% Output: Augmented spectra and corresponding phase correction values

clc;close all;clear;

% Set input and output directories
maindir ='data/samples/data_aug/topspin_formats_data';
savePath_spectra ='data/samples/data_aug/Extracted_spectra';
savePath_phase ='data/samples/data_aug/Extracted_phase';

% Create output directories
if ~exist(savePath_spectra, 'dir')
    mkdir(savePath_spectra);
end

if ~exist(savePath_phase, 'dir')
    mkdir(savePath_phase);
end

% Process all sample directories
subdir  = dir( maindir );
for i = 1:length( subdir )
    if( isequal( subdir( i ).name, '.' )||...
        isequal( subdir( i ).name, '..')||...
        startsWith(subdir( i ).name, '.') ||....
        ~subdir( i ).isdir)
        continue;
    end
    maindir1 = fullfile( maindir, subdir( i ).name);
    subdir1  = dir( maindir1 );
    for k = 1:length( subdir1 )
        if( isequal( subdir1( k ).name, '.' )||...
            isequal( subdir1( k ).name, '..')||...
            ~subdir( k ).isdir)
            continue;
        end
    fiddirpath = fullfile( maindir1, subdir1( k ).name);

%     fiddirpath = maindir1;

% Step 1: Load FID data
fidPath = [fiddirpath,'/fid'];
swHz = ReadTopspinParam(fidPath, 'SW_h');
fidpoints = ReadTopspinParam(fidPath, 'TD');
SizeTD1 = 1;
ByteOrder = 2;
[fid, SizeTD2, SizeTD1] = GetFIdFromBidary(fidPath, fidpoints, SizeTD1, ByteOrder);

% Read spectral parameters
specPath = [fiddirpath ,'/pdata/1/1r'];
swppm = ReadTopspinParam(specPath, 'SW');
offsetppm = ReadTopspinParam(specPath, 'OFFSET');
lockppm = ReadTopspinParam(specPath, 'LOCKPPM');
swppmHalf=swppm/2;
IdealWater=offsetppm-swppmHalf;
Diff=-(IdealWater-lockppm);

% Step 2: Digital filter correction
DECIM = ReadTopspinParam(fidPath, 'DECIM');
DSPFVS = ReadTopspinParam(fidPath, 'DSPFVS');
DIGMOD = ReadTopspinParam(fidPath, 'DIGMOD');
GRPDLY = ReadTopspinParam(fidPath, 'GRPDLY');
NrPointsToShift = DetermineBrukerDigitalFilter(DECIM, DSPFVS, DIGMOD,GRPDLY);
ShiftNum =  round( NrPointsToShift );
ShiftResidual= NrPointsToShift-ShiftNum;

% Step 3: Apply window function
lb = ReadTopspinParam(specPath, 'LB');
fidSize=length(fid);
points = 0:1:(fidSize-1);
t=exp(-points.*(pi*lb/swHz));
WindowFidData=fid.*t;

% Step 4: Zero padding
ftSize = ReadTopspinParam(specPath, 'SI');
fidAddWin = [WindowFidData zeros(1,ftSize-fidSize)];

% Step 5: Digital filter delay correction
TempFidData=fidAddWin(:,1:ShiftNum);
FidData=[fidAddWin(:, ShiftNum+1:end) TempFidData];

% Step 6: FFT and phase correction
DataBeforePhase1 = ifftshift((ifft(FidData(1,:))));
PHC0_orig = ReadTopspinParam(specPath, 'PHC0');
PHC1_orig = ReadTopspinParam(specPath, 'PHC1');
PHC0_orig = PHC0_orig*(pi/180.0);
PHC1_orig = PHC1_orig*(pi/180.0)+ShiftResidual*360*(pi/180.0);
x = ((0:length(DataBeforePhase1)-1))/length(DataBeforePhase1);

PhaseDataAfterphc = DataBeforePhase1 .* exp(-1i*(PHC0_orig + PHC1_orig * x));

% Step 7: Data augmentation - generate random phase variations
numbers = 140;  % Number of augmented samples per spectrum
Ph0_expand = zeros(size(numbers));
Ph1_expand = zeros(size(numbers));
DataBeforePhase2 = zeros(numbers,length(DataBeforePhase1));
DataBeforePhase2_corr = zeros(numbers,length(DataBeforePhase1));
gt_ph0_ph1 = zeros(numbers,2);

% Generate augmented samples with random phase errors
for inum = 1:numbers
    % Generate random phase parameters
    p1 = 0+180*rand(1,1);  % PHC0: 0-180 degrees
    p2 = -40+80*rand(1,1); % PHC1: -40 to +40 degrees
    Ph0_expand(inum) = p1 * pi / 180;
    Ph1_expand(inum) = p2 * pi / 180;

    % Apply random phase error
    DataBeforePhase2(inum,:) = PhaseDataAfterphc .* exp(1i * (Ph1_expand(inum) * x + Ph0_expand(inum)));
    DataBeforePhase2_corr(inum,:)=DataBeforePhase2(inum, :).*exp(-1i*(Ph1_expand(inum) * x + Ph0_expand(inum)));
    
    % Store ground truth phase values
    gt_ph0_ph1(inum, 1) = Ph0_expand(inum) * 180 / pi;
    gt_ph0_ph1(inum, 2) = Ph1_expand(inum) * 180 / pi;  
    
    % Generate sample name
    an = strsplit(fiddirpath, filesep);
    str1 = char(an{end-1});
    str2 = char(an{end});
    A = [str1, '_', str2];

    % Save augmented spectrum
    data_complex = DataBeforePhase2(inum, :);
    data_real_row = real(data_complex);
    data_imag_row = imag(data_complex);
    filename = fullfile(savePath_spectra, sprintf('%s_%d.txt', A, inum));
    fid = fopen(filename, 'w');
    fprintf(fid, '%f%+fi\n', [data_real_row; data_imag_row]);
    fclose(fid);

    % Save corresponding phase parameters
    data_gt_ph0_ph1 = gt_ph0_ph1(inum,:); 
    filename = fullfile(savePath_phase, sprintf('%s_%d.txt', A, inum));  
    fid = fopen(filename, 'w');
    fprintf(fid, '%f ', data_gt_ph0_ph1);
    fprintf(fid, '\n');  
    fclose(fid);    
end
    end
disp('Processing completed')
end