function [SpecDataReal,SpecDataImg] = LoadBrukerSpec(subdirpath)
%LOADBRUKERSPEC Summary of this function
%   Loads Bruker spectral data, where pdata\*\1r is the real part and pdata\*\1i is the imaginary part

fnamereal = fullfile( subdirpath, '\pdata\1\1r' );
fnameimg = fullfile( subdirpath, '\pdata\1\1i' );
% fidpoints = 32768; %naojiye
fidpoints = 65536; %niaoye CPMG
% fidpoints = 131072; %Noesy
SizeTD1 = 1;
ByteOrder = 2;

[ SpecDataReal ] = GetBrukerSpecfromBinary(fnamereal, SizeTD1, fidpoints, ByteOrder);
[ SpecDataImg ] = GetBrukerSpecfromBinary(fnameimg, SizeTD1, fidpoints, ByteOrder);
%Plot for direct visualization
% SpecDataReal(16000:16500)=0;
% figure(1);
% plot(SpecDataReal);
% figure(2);
% plot(SpecDataImg);

end

