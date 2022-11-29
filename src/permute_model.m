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

% Avoid to show figures since they will slowdown the execution.
set(0,'DefaultFigureVisible','off')

% Estimate the non-null model.
non_null_model = outer_cv(X, y, model);
[measures, ~] = metricsnplots(model.RegressionType, non_null_model);

% Take proper performance measure based on the type of target.
if strcmp(model.RegressionType, 'normal')
    non_null_stat = measures.Rsq;
else
    non_null_stat = measures.BalanceAccuracy;
end

for it = 1:iterations
    
    % Permute the target.
    perm_idx = randperm(nsubj);
    y = y(perm_idx);
    
    % Estimate the null model.
    null_model = outer_cv(X, y, model);
    [measures, ~] = metricsnplots(model.RegressionType, null_model);
    
    % Take proper performance measure based on the type of target.
    if strcmp(model.RegressionType, 'normal')
        null_stat(it) = measures.Rsq;
    else
        null_stat(it) = measures.BalanceAccuracy;
    end
end

    % Calculate the p-value.
    pval = sum(abs(null_stat) >= abs(non_null_stat)) / iterations;
    
    set(0,'DefaultFigureVisible','on')
end