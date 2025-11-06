%% PDATA2TXT_DATAAUG - NMR Data Augmentation for Training (Robust Version)
%
% Processes Bruker NMR data and generates augmented training samples
% by applying random phase corrections to create multiple variants
% of each spectrum for deep learning model training.
%
% Output: Augmented spectra and corresponding phase correction values
%

clc;close all;clear;

% Set input and output directories
maindir ='F:\cg\download\PD-RAN-main\data\samples\topspin_formats_data\Metabolomics';
savePath_spectra ='F:\cg\download\PD-RAN-main\data\samples\data_aug\input_spectra';
savePath_phase ='F:\cg\download\PD-RAN-main\data\samples\data_aug\gt_phase';

% Create output directories
if ~exist(savePath_spectra, 'dir')
    mkdir(savePath_spectra);
end

if ~exist(savePath_phase, 'dir')
    mkdir(savePath_phase);
end

% Process all sample directories
fid_list = {};

% Implement recursion using a stack or queue
dir_stack = {maindir};

while ~isempty(dir_stack)
    current_dir = dir_stack{end};
    dir_stack(end) = [];

    subdir = dir(current_dir);

for i = 1:numel(subdir)
    if subdir(i).isdir && ...
       ~startsWith(subdir(i).name, '.') && ...
       ~isequal(subdir(i).name, '.') && ...
       ~isequal(subdir(i).name, '..')

        this_dir = fullfile(current_dir, subdir(i).name);

        % Check if this is an FID directory
        files = dir(this_dir);
        names = {files.name};
        if any(strcmp(names, 'fid')) || any(strcmp(names, 'acqus'))
            fiddirpath = this_dir;
            fprintf('Find the FID directory: %s\n', fiddirpath);
            fid_list{end+1} = fiddirpath;

        % Step 1: Load FID data
            fidPath = [fiddirpath,'/fid'];
            acquPath = [fiddirpath,'/acqus']; % NEW: Define acqus file path for reading parameters.

            swHz = ReadTopspinParam(acquPath, 'SW_h');
            fidpoints = ReadTopspinParam(acquPath, 'TD');
            SizeTD1 = 1;
            
        %% Automatically determine Byte Order from the acqus file
            % This is the key fix to handle both old and new data formats.
            BYTORDA_val = ReadTopspinParam(acquPath, 'BYTORDA'); % Read the byte order parameter from acqus file.
            if (BYTORDA_val == 0)
                ByteOrder = 2; % Set to 2 for little-endian (modern PC-based systems).
                disp('BYTORDA = 0 detected. Using Little-Endian byte order.');
            else
                ByteOrder = 1; % Set to 1 for big-endian (legacy workstation systems).
                disp('BYTORDA is not 0. Using Big-Endian byte order.');
            end
        
            [fid, SizeTD2, SizeTD1] = GetFIdFromBidary(fidPath, fidpoints, SizeTD1, ByteOrder);

        %% Read spectral parameters
            %% NEW: Dynamically find the processing number (procno)
            pdataPath = fullfile(fiddirpath, 'pdata');
            procno = ''; % Initialize procno as empty
            if exist(pdataPath, 'dir')
                pdata_contents = dir(pdataPath);
                for k = 1:length(pdata_contents)
                    % Find the first subdirectory which is a number
                    if pdata_contents(k).isdir && ~isnan(str2double(pdata_contents(k).name))
                        procno = pdata_contents(k).name;
                        fprintf('Found processing directory: pdata/%s\n', procno);
                        break; % Stop after finding the first one
                    end
                end
            end

            if isempty(procno)
                fprintf('ERROR: Could not find a valid processing directory inside %s. Skipping sample.\n', pdataPath);
                continue; % Skip to the next sample in the loop
            end

            % Construct the correct specPath using the detected procno
            specPath = fullfile(fiddirpath, 'pdata', procno, '1r');
            specPath = strrep(specPath, '\', '/');
           
            swppm = ReadTopspinParam(specPath, 'SW');
            offsetppm = ReadTopspinParam(specPath, 'OFFSET');
            lockppm = ReadTopspinParam(acquPath, 'LOCKPPM'); 
            swppmHalf=swppm/2;
            IdealWater=offsetppm-swppmHalf;
            Diff=-(IdealWater-lockppm);

        % Step 2: Digital filter correction
            DECIM = ReadTopspinParam(acquPath, 'DECIM'); 
            DSPFVS = ReadTopspinParam(acquPath, 'DSPFVS');
            DIGMOD = ReadTopspinParam(acquPath, 'DIGMOD');
            GRPDLY = ReadTopspinParam(acquPath, 'GRPDLY');
            NrPointsToShift = DetermineBrukerDigitalFilter_Mod(DECIM, DSPFVS, DIGMOD,GRPDLY);
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
            if(ftSize~=64*1024)
                ftSize=64*1024;
            end
            fidAddWin = [WindowFidData zeros(1,ftSize-fidSize)];

        % Step 5: Digital filter delay correction
            TempFidData=fidAddWin(:,1:ShiftNum);
            FidData=[fidAddWin(:, ShiftNum+1:end) TempFidData];

        % Step 6: FFT and phase correction
            DataBeforePhase1 = ifftshift((ifft(FidData(1,:))));
            PHC0_orig = ReadTopspinParam(specPath, 'PHC0');
            PHC1_orig = ReadTopspinParam(specPath, 'PHC1');
            PHC0_orig = PHC0_orig*(pi/180.0);
            PHC1_orig_Shift = PHC1_orig*(pi/180.0)+ShiftResidual*360*(pi/180.0);
            
            x = ((0:length(DataBeforePhase1)-1))/length(DataBeforePhase1);

            PhaseDataAfterphc = DataBeforePhase1 .* exp(-1i*(PHC0_orig + PHC1_orig_Shift * x));
            
            figure();plot(real(DataBeforePhase1), 'b');hold on;plot(real(PhaseDataAfterphc), 'r');
            
        % Step 7: Data augmentation - generate random phase variations
            numbers = 5;  % Number of augmented samples per spectrum
            Ph0_expand = zeros(size(numbers));
            Ph1_expand = zeros(size(numbers));
            DataBeforePhase2 = zeros(numbers,length(DataBeforePhase1));
            DataBeforePhase2_corr = zeros(numbers,length(DataBeforePhase1));
            gt_ph0_ph1 = zeros(numbers,2);

        % Generate augmented samples with random phase errors
            for inum = 1:numbers
                % Generate random phase parameters
                p1 = -180 + 360 * rand(1,1);
                p2 = -40 + 80 * rand(1,1);
                Ph0_expand(inum) = p1 * pi / 180;
                Ph1_expand(inum) = p2 * pi / 180;

                % Apply random phase error
                DataBeforePhase2(inum,:) = PhaseDataAfterphc .* exp(1i * (Ph1_expand(inum) * x + Ph0_expand(inum)));
                DataBeforePhase2_corr(inum,:)=DataBeforePhase2(inum, :).*exp(-1i * (Ph1_expand(inum) * x + Ph0_expand(inum)));

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
            else
                dir_stack{end+1} = this_dir;
            end
        end
    end
end
disp('Processing completed')