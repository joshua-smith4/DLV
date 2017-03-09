function rGTData = TSD_readGTData(aGTFile)
% function rGTData = TSD_readGTData(aGTFile)
% 
% *************************************************************************
% Author:   Sebastian Houben
%           University of Bochum
% Mail:     tsd-benchmark@ini.rub.de
% Date:     01.12.2012
% *************************************************************************
%
% *************************************************************************
% Description
% *************************************************************************
%
% This function can be applied to facilitate working with the dataset from
% the "German Traffic Sign Detection Benchmark"
% (http://benchmark.ini.rub.de). 
% 
% The function reads a ground truth file that comes along with the download
% package of the training data. It returns structs with the read
% information.
%
% aGTFile   full name of the ground truth file to be read
% rGTData   array of structs containing the ground truth
%           .fileNo     Number of the file the ground truth region of
%           interrest (ROI) accounts for
%           .leftCol
%           .topRow
%           .rightCol
%           .bottomRow  ROI that pinpoints a traffic sign's location in the image with
%           fileNo
%           .class      a string describing the kind of traffic sign
%           .classID    a number describing the kind of traffic sign (for a
%           listing of the used IDs refer to the ReadMe.txt in the download
%           package)
%           .category   category the traffic sign belongs to
%           ('prohibitory', 'mandatory', 'danger', 'other')
%
% see also TSD_testMyDetector, TSD_testMySubmissionFile


fID = fopen(aGTFile, 'r');

rGTData = [];
data = fscanf(fID, '%05d.ppm;%d;%d;%d;%d;%d');
data = reshape(data, [6, numel(data) / 6])';

prohibitoryClassIds = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16];
mandatoryClassIds = [33, 34, 35, 36, 37, 38, 39, 40];
dangerClassIds = [11, 18, 19, 20 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31];
% prohibitoryClassIds = [17, 16, 15, 10, 9, 8, 7, 5, 4, 3, 2, 1, 0];
% mandatoryClassIds = [33, 34, 35, 36, 37, 38, 39, 40];
% dangerClassIds = [11, 13, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31];

for i = 1:size(data, 1)
    rGTData(i).fileNo = data(i, 1);
    rGTData(i).leftCol = data(i, 2);
    rGTData(i).topRow = data(i, 3);
    rGTData(i).rightCol = data(i, 4);
    rGTData(i).bottomRow = data(i, 5);
    rGTData(i).class = '';
    rGTData(i).classID = data(i, 6);
    if ismember(rGTData(i).classID, prohibitoryClassIds)
        rGTData(i).category = 'prohibitory';
    elseif ismember(rGTData(i).classID, mandatoryClassIds)
        rGTData(i).category = 'mandatory';
    elseif ismember(rGTData(i).classID, dangerClassIds)
        rGTData(i).category = 'danger';
    else
        rGTData(i).category = 'other';
    end
end
    
