function [DataBeforePhase1,PHC0,PHC1] = LoadBrukerFID_addphase(dataPath, PHC0, PHC1)
%LOADBRUKERFID_ADDPHASE Load Bruker FID data and apply phase correction
%   This function loads Bruker FID data from the specified path, applies
%   window function, performs Fourier transform, and applies phase correction
%   with the provided PHC0 and PHC1 values.
%
%   Inputs:
%     dataPath - Path to the Bruker data directory
%     PHC0 - Zero-order phase correction value (in degrees)
%     PHC1 - First-order phase correction value (in degrees)
%
%   Outputs:
%     DataBeforePhase1 - Data after Fourier transform but before phase correction
%     PHC0 - Applied zero-order phase correction (in degrees)
%     PHC1 - Applied first-order phase correction (in degrees)

pDataDir = [dataPath '/pdata/1'];

%% Read FID
fidPath = [dataPath,'/fid'];
swHz = ReadTopspinParam(fidPath, 'SW_h');
fidpoints = ReadTopspinParam(fidPath, 'TD');
SizeTD1 = 1;
ByteOrder = 2;
[fid, SizeTD2, SizeTD1] = GetFIdFromBidary(fidPath, fidpoints, SizeTD1, ByteOrder);
%figure(1);
%plot(real(fid(1,:)));
%title('raw FID');

specPath = [pDataDir '/1r'];  %% Parameters are recorded in procs file
swppm = ReadTopspinParam(specPath, 'SW');
offsetppm = ReadTopspinParam(specPath, 'OFFSET');
lockppm = ReadTopspinParam(specPath, 'LOCKPPM');
swppmHalf=swppm/2;
IdealWater=offsetppm-swppmHalf;
Diff=-(IdealWater-lockppm);

%% Read digital filter parameters
DECIM = ReadTopspinParam(fidPath, 'DECIM');
DSPFVS = ReadTopspinParam(fidPath, 'DSPFVS');
DIGMOD = ReadTopspinParam(fidPath, 'DIGMOD');
GRPDLY = ReadTopspinParam(fidPath, 'GRPDLY');
NrPointsToShift = DetermineBrukerDigitalFilter(DECIM, DSPFVS, DIGMOD,GRPDLY);
ShiftNum =  round( NrPointsToShift );
ShiftResidual= NrPointsToShift-ShiftNum;

%% Apply window function
lb = ReadTopspinParam(specPath, 'LB');
fidSize=length(fid);
points = 0:1:(fidSize-1);
t=exp(-points.*(pi*lb/swHz));
WindowFidData=fid.*t;


%% Zero-filling
ftSize = ReadTopspinParam(specPath, 'SI');
fidAddWin = [WindowFidData zeros(1,ftSize-fidSize)];


%% Digital filter correction
TempFidData=fidAddWin(:,1:ShiftNum);
FidData=[fidAddWin(:, ShiftNum+1:end) TempFidData];

%% Inverse Fourier transform
DataBeforePhase1 = ifftshift((ifft(FidData(1,:))));

% x = ((0:length(DataBeforePhase1)-1))/length(DataBeforePhase1);
figure(1);
plot(real(DataBeforePhase1));title('Unphased(real part)');

%% Phase correction
DataSize = length(DataBeforePhase1);
PHC0=PHC0*(pi/180.0);
PHC1=PHC1*(pi/180.0);
disp(PHC0*180/pi);
disp(PHC1*180/pi);
a_num = ((0:DataSize-1))/(DataSize);
PhaseDataAfterphc = DataBeforePhase1 .* exp(-1i*(PHC0+PHC1*a_num));
PhaseDataAfterphc = PhaseDataAfterphc/max(abs(PhaseDataAfterphc))*65536;
figure(2);
plot(real(PhaseDataAfterphc));title('Phased(real part)');
disp(PHC0);
disp(PHC1);
figure(3);
plot(imag(PhaseDataAfterphc));title('Phased(imag part)');

%% Save phase-corrected data to pdata/1
Save2Bruker(PhaseDataAfterphc,pDataDir);


end

