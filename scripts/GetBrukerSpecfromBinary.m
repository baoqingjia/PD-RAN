function [ SpecData ] = GetBrukerSpecfromBinary(fname,SizeTD1, SizeTD2, ByteOrder)
%LOADBRUKERSPEC Summary of this function goes here
%   Detailed explanation goes here
if (ByteOrder == 2)
    id=fopen(fname, 'r', 'l');			%use little endian format if asked for
else
    id=fopen(fname, 'r', 'b');			%use big endian format by default
end

SpecData=zeros(1,SizeTD1*SizeTD2);

 [SpecData, count] = fread(id, SizeTD1*SizeTD2, 'int');
 
 fclose(id);
end

