function [rData, rHeadLineData] = io_readTxtFile(aFileName, aFormat, aHeadLineFormat)
% function [rData, rHeadLineData] = io_readTxtFile(aFileName, aFormat[, aHeadLineFormat / aSkipLines])
% *************************************************************************
% Autor:    Sebastian Houben
%           Ruhr-Universität Bochum, Tel. 0234 - 32 - 25566
% Mail:     sebastian.houben@ini.rub.de
% Datum:    12.09.2012
% *************************************************************************
%
% *************************************************************************
% Beschreibung
% *************************************************************************
%
% Liest ein mehrdimensionales Array aus einer Textdatei.
%
% aFileName     voller Name der zu lesenden Textdatei
% aFormat       zeilenweises Format in fprintf-Darstellung
% aHeadLineFormat   (optional) die erste Zeile darf sich vom Rest unterscheiden und
%                   ein anderes Format besitzen, falls nicht angegeben,
%                   wird keine Headerzeile erwartet
% aSkipLines        (optional, statt aHeadlineFormat) Anzahl der Zeilen,
%                   die zu Anfang der Datei übersprungen werden
%
% rData         die Daten, die aus der Textdatei gelesen wurden
% rHeadLineData die Daten, die aus der Headerzeile gelesen wurden

if aFormat(end) ~= '\n'
    aFormat(end + 1:end + 2) = '\n';
end

[fID, msg] = fopen(aFileName, 'r');

if strcmp(msg, '') ~= 1
    error(['Fehler beim Oeffnen von ', aFileName, ':  ', msg]);
end

if exist('aHeadLineFormat', 'var')
    if ischar(aHeadLineFormat)
        sLine = fgetl(fID);
        rHeadLineData = sscanf(sLine, aHeadLineFormat);
    else % aSkipLInes
        aSkipLines = aHeadLineFormat;
        for i = 1:aSkipLines
            fgetl(fID);
        end
    end
else
    rHeadLineData = [];
end

numElemsPerLine = numel(strfind(aFormat, '%'));

rData = fscanf(fID, aFormat);
rData = reshape(rData, [numElemsPerLine, numel(rData) / numElemsPerLine])';

fclose(fID);

