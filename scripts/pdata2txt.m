%LOAD BRUKER FID
clc;close all;clear;

maindir ='D:\MySync\Matlab Data\PD-RAN\Data\samples\topspin_formats_data';
savePath_spectra ='D:\MySync\Matlab Data\PD-RAN\Data\samples\Extracted_spectra';
savePath_phase ='D:\MySync\Matlab Data\PD-RAN\Data\samples\Extracted_phase';

if ~exist(savePath_spectra, 'dir')
    mkdir(savePath_spectra);
end

if ~exist(savePath_phase, 'dir')
    mkdir(savePath_phase);
end

subdir  = dir(maindir);

for i = 1:length( subdir )
    if( isequal( subdir( i ).name, '.' )||...
        isequal( subdir( i ).name, '..')||... 
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

%% loadFID
    fidPath = [fiddirpath,'/fid'];
    swHz = ReadTopspinParam(fidPath, 'SW_h');
    fidpoints = ReadTopspinParam(fidPath, 'TD');
    SizeTD1 = 1;
    ByteOrder = 2;
    [fid, SizeTD2, SizeTD1] = GetFIdFromBidary(fidPath, fidpoints, SizeTD1, ByteOrder);

    specPath = [fiddirpath ,'/pdata/1/1r'];
    swppm = ReadTopspinParam(specPath, 'SW');
    offsetppm = ReadTopspinParam(specPath, 'OFFSET');
    lockppm = ReadTopspinParam(specPath, 'LOCKPPM');
    swppmHalf=swppm/2;
    IdealWater=offsetppm-swppmHalf;
    Diff=-(IdealWater-lockppm);
%% Get the number of points to move
    DECIM = ReadTopspinParam(fidPath, 'DECIM');
    DSPFVS = ReadTopspinParam(fidPath, 'DSPFVS');
    DIGMOD = ReadTopspinParam(fidPath, 'DIGMOD');
    GRPDLY = ReadTopspinParam(fidPath, 'GRPDLY');
    NrPointsToShift = DetermineBrukerDigitalFilter(DECIM, DSPFVS, DIGMOD,GRPDLY);
    ShiftNum =  round( NrPointsToShift );
    ShiftResidual= NrPointsToShift-ShiftNum;
%% window function
    lb = ReadTopspinParam(specPath, 'LB');
    fidSize=length(fid);
    points = 0:1:(fidSize-1);
    t=exp(-points.*(pi*lb/swHz));
    WindowFidData=fid.*t;
%% zero padding
    ftSize = ReadTopspinParam(specPath, 'SI');
    if(ftSize~=64*1024)
        ftSize=64*1024;
    end
    fidAddWin = [WindowFidData zeros(1,ftSize-fidSize)];
%% point shifting 
    TempFidData=fidAddWin(:,1:ShiftNum);
    FidData=[fidAddWin(:, ShiftNum+1:end) TempFidData];
%% fft
    DataBeforePhase1 = ifftshift((ifft(FidData(1,:))));
    PHC0_orig = ReadTopspinParam(specPath, 'PHC0');
    PHC1_orig = ReadTopspinParam(specPath, 'PHC1');
    PHC0_orig = PHC0_orig*(pi/180.0);
    PHC1_orig = PHC1_orig*(pi/180.0)+ShiftResidual*360*(pi/180.0);
    x = ((0:length(DataBeforePhase1)-1))/length(DataBeforePhase1);
    PhaseDataAfterphc = DataBeforePhase1 .* exp(-1i*(PHC0_orig+PHC1_orig*x));
    
    gt_ph0_ph1 = zeros(1,2);
    gt_ph0_ph1(:, 1) = PHC0_orig*(180.0/pi);
    gt_ph0_ph1(:, 2) = PHC1_orig*(180.0/pi); 
    
%     if gt_ph0_ph1(:, 1) <= 180

        an = strsplit(fiddirpath,'\');
        str1 = char(an(length(an)-1)); 
        str2 = char(an(length(an))); 
        A = [str1, '_', str2];

        data_real_row = real(DataBeforePhase1);
        data_imag_row = imag(DataBeforePhase1);
        filename = fullfile(savePath_spectra, sprintf('%s.txt', A));
        fid = fopen(filename, 'w');
        fprintf(fid, '%f%+fi\n', [data_real_row; data_imag_row]);  
        fclose(fid);

        filename = fullfile(savePath_phase, sprintf('%s.txt', A));  
        fid = fopen(filename, 'w');
        fprintf(fid, '%f ', gt_ph0_ph1);
        fprintf(fid, '\n');  
        fclose(fid);  
%     end

    disp('over')
    end
end