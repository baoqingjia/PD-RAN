function  [fid, SizeTD2, SizeTD1]= GetFIdFromBidary(fname, SizeTD2, SizeTD1, ByteOrder)
%  get fid of Bruker data from file
%  Input 
%         fname:   the file name of varian data 
%         SizeTD2: the size of 2D points
%         SizeTD1: the size of 1D points
%         ByteOrder: big endian or little endian
%  Output
%         fid: the fid of varian data
%         SizeTD2: the size of 2D points
%         SizeTD1: the size of 1D points

if (ByteOrder == 2)
    id=fopen(fname, 'r', 'l');			%use little endian format if asked for
else
    id=fopen(fname, 'r', 'b');			%use big endian format by default
end
a = zeros(SizeTD2 * SizeTD1, 1);
[a, count] = fread(id, SizeTD2 * SizeTD1, 'int32');

for tel = 1:SizeTD1
    fid(tel, 1:(SizeTD2 / 2)) = (a((tel - 1) * SizeTD2 + (1 : 2 : SizeTD2 - 1)) + sqrt(-1) * a((tel - 1) * SizeTD2 + (2 : 2 : SizeTD2))).';
end
fclose(id);