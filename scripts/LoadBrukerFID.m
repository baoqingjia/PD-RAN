function [DataBeforePhase1] = LoadBrukerFID(fname)

fidpoints = 16384;
SizeTD1 = 1;
ByteOrder = 2;
[fid2, SizeTD2, SizeTD1] = GetFIdFromBidary(fname, fidpoints, SizeTD1, ByteOrder);
fidpoints2 = 24576;
fid=cat(2,fid2,zeros(1,fidpoints2));%new
% figure(1);
% plot(real(fid(1,:)));
% figure(2);
% plot(real(fid(2,:)));

DECIM = 2496;
DSPFVS = 20;
DIGMOD = 1;
GRPDLY = 67.9842376708984;
NrPointsToShift = DetermineBrukerDigitalFilter(DECIM, DSPFVS, DIGMOD,GRPDLY);

L = length(fid);
ShiftNum = floor( NrPointsToShift );
TempFidData=fid(:,1:ShiftNum);
FidData=[fid(:, ShiftNum+1:L) TempFidData];

DataBeforePhase1 = fftshift((ifft(FidData(1,:))));


end

