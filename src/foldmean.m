function[matmean] = foldmean(xfoldc)
% Compute mean across the folders of the AUC.
% Inputs:
%   xfoldc: folders.
% Outups:
%   matmean: mean across the folders.

nrows = zeros(size(xfoldc, 2), 1);

for i = 1:size(xfoldc, 2)
    nrows(i, 1) = size(xfoldc{i}, 1);
end

maxrow = max(nrows);

for i = 1:size(xfoldc, 2)
    sinsize = size(xfoldc{i}, 1);
    if size(xfoldc{i}, 1) ~= maxrow 
    xfoldc{i}(sinsize : maxrow, 1) = NaN;
    end
end

matfold = cell2mat(xfoldc);
matmean = nanmean(matfold, 2);
end