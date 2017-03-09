function TSD_testMySubmissionFile()
% function TSD_testMySubmissionFile()
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
% detector function by checking a given text file containing results.
% You will have to adapt the first few lines with parameters that fit your needs.
% The text file is supposed to have the same format as needed for
% submission during the online competition phase, so you can also use this
% function to check your submission file format.
%
% see also TSD_readGTData, TSD_testMyDetector

% to use this function you will have to adjust the following parameters
benchmarkPath = 'D:\rtcv_data\TrafficSignDetection\TSD_Benchmark\TrainIJCNN2013';    % path you extracted the dataset to
category = 'prohibitory'; % the traffic sign category your submission is aimed at 'prohibitory', 'mandatory', or 'danger' 
submissionFile = 'D:\rtcv_data\TrafficSignDetection\TSD_Benchmark\TrainIJCNN2013\ex.txt'; % full name of the submission file you want to test
verbose = false;     % defines whether or not a detailed message for every detection is printed to the default output





% % % START OF EVALUATION (DO NOT CHANGE BELOW)

gtData = TSD_readGTData([benchmarkPath, '\gt.txt']);
gtData = gtData( strcmp({gtData(:).category}, category) );

fID = fopen(submissionFile);
submissionData = fscanf(fID, '%05d.ppm;%d;%d;%d;%d');
submissionData = reshape(submissionData, [5, numel(submissionData) / 5])';
fclose(fID);

jaccCoefThresh = 0.6;

TP = 0;
FP = 0;
FN = 0;

for imgNum = 0:599
    currSubmissionData = submissionData(submissionData(:, 1) == imgNum, :);
    leftCols = currSubmissionData(:, 2);
    topRows = currSubmissionData(:, 3);
    rightCols = currSubmissionData(:, 4);
    bottomRows = currSubmissionData(:, 5);
    
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
        maxJaccCoeff = jaccCoefThresh;
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
TP
FP
FN