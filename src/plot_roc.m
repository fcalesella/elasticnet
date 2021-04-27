function [legends] = plot_roc(xfold, yfold, k)
% Plot the ROC for each AUC folder.
% Inputs:
%   xfold: folders of the predictors.
%   yfold: folders of the targets.
%   k: number of folders.
% Outputs:
%   legends: legend with folder number.

legends = cell(1, k);
for i = 1:k
    legends{i} = sprintf('Fold %i', i);
    plot(xfold{i}, yfold{i})
    hold on
end
end