function TSD_testMyDetector()
% function TSD_testMyDetector()
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
% The function determines the performance (precision and recall) of a given
% detector function. You will have to adapt the first few lines with
% parameters that fit your needs. 
%
% see also TSD_readGTData, TSD_testMySubmissionFile

% to use this function you will have to adjust the following parameters
benchmarkPath = 'D:\rtcv_data\TrafficSignDetection\TSD_Benchmark\TrainIJCNN2013';    % path you extracted the dataset to
category = 'prohibitory'; % the traffic sign category your submission is aimed at 'prohibitory', 'mandatory', or 'danger' 
detectorFunc = @myTestDetector; % a handle to your detector function
paramsForDetectorFunc = {[]}; % a cell array with additional parameters your function needs
verbose = false;     % defines whether or not a detailed message for every detection is printed to the default output

% your detector should be a Matlab function with 1 argument (the full image) and 4 return
% values (the detected ROIs as vectors of the left column, the right
% column, the top row, and the bottom row
% [rLeftCols, rRightCols, rTopRows, rBottomRows] = myDetectorFunc(aImg)


% % % START OF EVALUATION (DO NOT CHANGE BELOW)

gtData = TSD_readGTData([benchmarkPath, '\gt.txt']);
gtData = gtData( strcmp({gtData(:).category}, category) );

TP = 0;
FP = 0;
FN = 0;

for imgNum = 0:599
    currFileName = [benchmarkPath, '\', num2str(imgNum, '%05d'), '.ppm'];
    fullImage = imread(currFileName);
    
    [leftCols, rightCols, topRows, bottomRows] = detectorFunc( fullImage, paramsForDetectorFunc );
    paramsForDetectorFunc = {1};
    
    imgGtIdxs = ([gtData(:).fileNo] == imgNum);
    gtLeftCols = [gtData(imgGtIdxs).leftCol];
    gtRightCols = [gtData(imgGtIdxs).rightCol];
    gtTopRows = [gtData(imgGtIdxs).topRow];
    gtBottomRows = [gtData(imgGtIdxs).bottomRow];
    
    if verbose
        fprintf(1, 'Image %d:\n', imgNum);
    end
    
    gtSignHit = false(numel(gtLeftCols), 1);
    for roiIdx = 1:numel(leftCols)
        maxJaccCoeff = 0.6;
        maxGtRoiIdx = 0;
        for gtRoiIdx = 1:numel(gtLeftCols)
            jaccCoeff = bm_getJaccardCoefficient([leftCols(roiIdx), topRows(roiIdx), rightCols(roiIdx), bottomRows(roiIdx)], ...
                [gtLeftCols(gtRoiIdx), gtTopRows(gtRoiIdx), gtRightCols(gtRoiIdx), gtBottomRows(gtRoiIdx)]);
            if jaccCoeff > maxJaccCoeff
                maxJaccCoeff = jaccCoeff;
                maxGtRoiIdx = gtRoiIdx;
            end
        end
        if maxGtRoiIdx == 0
            FP = FP + 1;
            if verbose
                fprintf(1, 'Miss: cols=%d..%d, rows=%d..%d\n', leftCols(roiIdx), rightCols(roiIdx), topRows(roiIdx), bottomRows(roiIdx));
            end
        else
            gtSignHit(maxGtRoiIdx) = true;
            if verbose
                fprintf(1, 'Hit: cols=%d..%d, rows=%d..%d matches cols=%d..%d, rows=%d..%d\n', ...
                    leftCols(roiIdx), rightCols(roiIdx), topRows(roiIdx), bottomRows(roiIdx), ...
                    gtLeftCols(maxGtRoiIdx), gtRightCols(maxGtRoiIdx), gtTopRows(maxGtRoiIdx), gtBottomRows(maxGtRoiIdx));
            end
        end
    end
    
    TP = TP + sum(gtSignHit);
    FN = FN + sum(~gtSignHit);
    if verbose
        fprintf(1, 'Precision: %0.2f, Recall: %0.2f\n', TP / (TP + FP), TP / (TP + FN));
    end
end

fprintf(1, 'true positives = %d, false positives = %d, false negatives = %d\n', TP, FP, FN);
fprintf(1, 'Precision: %0.2f, Recall: %0.2f\n', TP / (TP + FP), TP / (TP + FN));