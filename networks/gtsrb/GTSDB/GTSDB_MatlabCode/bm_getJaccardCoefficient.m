function jacc = bm_getJaccardCoefficient( resRoi, gtRoi )
% function jacc = bm_getJaccardCoefficient( resRoi, gtRoi )
%
% roi = [ulx, uly, lrx, lry]

if resRoi(3) < gtRoi(1) || resRoi(4) < gtRoi(2) ...
        || resRoi(1) > gtRoi(3) || resRoi(2) > gtRoi(4)
    jacc = 0;
else
    intersectRoi(1) = max([resRoi(1), gtRoi(1)]);
    intersectRoi(2) = max([resRoi(2), gtRoi(2)]);
    intersectRoi(3) = min([resRoi(3), gtRoi(3)]);
    intersectRoi(4) = min([resRoi(4), gtRoi(4)]);

    if intersectRoi(3) < 0 || intersectRoi(4) < 0 
        intersectArea = 0;
    else
        intersectArea = (intersectRoi(3) - intersectRoi(1) + 1) * (intersectRoi(4) - intersectRoi(2) + 1);
    end

    unionArea = (resRoi(3) - resRoi(1) + 1) * (resRoi(4) - resRoi(2) + 1) + ...
        (gtRoi(3) - gtRoi(1) + 1) * (gtRoi(4) - gtRoi(2) + 1) - intersectArea;

    jacc = intersectArea / unionArea;
end
