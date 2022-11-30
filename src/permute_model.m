function [pval, null_stat] = permute_model(X, y, model, iterations)
% Perform random permutations of the model to assess if the model is
% significantly different from a null model.
% Inputs:
%   X: n-by-p data matrix, where n is the number of observations and p the
%       number of predictors.
%   y: target measure.
%   model: structure array containing the parameters to be passed to 
%       lassoglm.
%   iterations: integer indicating the number of permutations.
% Outputs:
%   pval: p-value as a result of the permutation test.
%   null_stat: the performance measure calculated from all the null models.

% Initialize.
[nsubj, ~] = size(X);
null_stat = zeros(iterations, 1);

% Estimate the non-null model.
non_null_model = outer_cv(X, y, model);

% Take proper performance measure based on the type of target.
if strcmp(model.RegressionType, 'normal')
    sst = sum((non_null_model.Tests - mean(non_null_model.Tests)).^2);
    sse = sum((non_null_model.Preds - non_null_model.Tests).^2);
    % Define R-squared as statistic.
    non_null_stat = 1 - sse / sst;
else
    % Compute confusion matrix and the derived measures (true poitives, 
    % true negatives, false negatives, and false positives).
    [~,cm,~,~] = confusion(non_null_model.Tests, non_null_model.Preds);
    TP = cm(2,2);
    TN = cm(1,1);
    FN = cm(2,1);
    FP = cm(1,2);
    sensitivity = TP/(TP + FN);
    specificity = TN/(TN+FP);
    % Define balance accuracy as statistic.
    non_null_stat = (sensitivity + specificity)/2; 
end

for it = 1:iterations
    
    % Permute the target.
    perm_idx = randperm(nsubj);
    y = y(perm_idx);
    
    % Estimate the null model.
    null_model = outer_cv(X, y, model);
    
    % Take proper performance measure based on the type of target.
    if strcmp(model.RegressionType, 'normal')
        sst = sum((null_model.Tests - mean(null_model.Tests)).^2);
        sse = sum((null_model.Preds - null_model.Tests).^2);
        % Define R-squared as statistic.
        null_stat(it) = 1 - sse / sst;
    else
        % Compute confusion matrix and the derived measures (true poitives, 
        % true negatives, false negatives, and false positives).
        [~,cm,~,~] = confusion(null_model.Tests, null_model.Preds);
        TP = cm(2,2);
        TN = cm(1,1);
        FN = cm(2,1);
        FP = cm(1,2);
        sensitivity = TP/(TP + FN);
        specificity = TN/(TN+FP);
        % Define balance accuracy as statistic.
        null_stat(it) = (sensitivity + specificity)/2; 
    end
end

    % Calculate the p-value.
    pval = (1 + sum(abs(null_stat) >= abs(non_null_stat))) / (1 + iterations);
    
end