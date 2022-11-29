function [cv_results] = outer_cv(X, y, cv_options)
% Perform nested cross-validation (outer loop).
% Inputs:
%   X: n-by-p data matrix, where n is the number of observations and p the
%       number of predictors.
%   y: target measure.
%   cv_options: structure array containing the parameters to be passed to
%       lassoglm.
% Outputs:
%   cv_results: structure array containing the nested cross-validation 
%       results.

% Create train and test partitions.
if strcmp(cv_options.RegressionType, 'normal')
    c = cvpartition(length(y), 'KFold', cv_options.KOuter);
else
    c = cvpartition(y,'KFold', cv_options.KOuter);
end

% Initialize the variables to be saved.
cv_coef = zeros(size(X, 2) + 1, cv_options.KOuter);
cv_lambda = zeros(1, cv_options.KOuter);
cv_alpha = zeros(1, cv_options.KOuter);
cv_auc = zeros(1, cv_options.KOuter);
cv_pred = [];
cv_test = [];
% cell array with every xfold (i.e. coordinate of curve ROC)
xfoldc = cell(1, cv_options.KOuter);
% cell array with every yfold (i.e. coordinate of curve ROC)
yfoldc = cell(1, cv_options.KOuter);

% Copy X and y to each worker.
X = parallel.pool.Constant(X);
y = parallel.pool.Constant(y);

parfor i = 1:cv_options.KOuter

    % Select train and test sets.
    idxTrain = training(c,i);
    idxTest = test(c, i);
    XTrain = X.Value(idxTrain,:);
    yTrain = y.Value(idxTrain);
    XTest = X.Value(idxTest,:);
    yTest = y.Value(idxTest);

    % Fit the model optimizing the alpha and the lambda hyper-parameters.
    [coef, best_alpha, best_lambda] = partuner(XTrain,yTrain,cv_options);

    cv_coef(:, i) = coef;
    cv_lambda(i) = best_lambda;
    cv_alpha(i) = best_alpha;

    if strcmp(cv_options.RegressionType, 'binomial')
        % For classification: compute the predictions on the test set,
        % binerize the predictions and calculate the AUC.
        yPred_con = glmval(coef,XTest,'logit');
        yPred = double (yPred_con>=0.5);
        [Xfold,Yfold,~,AUC] = perfcurve(yTest,yPred_con,'1');
        cv_auc(i) = AUC;
        xfoldc{i} = Xfold;
        yfoldc{i} = Yfold;
    else
        % For linear regression: compute the predictions on test set.
        yPred = XTest * coef(2:end) + coef(1);
    end

    cv_pred = [cv_pred, yPred.'];
    cv_test = [cv_test, yTest'];
end

% Save the results in a structure array.
cv_results = cv_options;
cv_results.Coef = cv_coef;
cv_results.Lambda = cv_lambda;
cv_results.Alpha = cv_alpha;
cv_results.AUC = cv_auc;
cv_results.XFold = xfoldc;
cv_results.YFold = yfoldc;
cv_results.Preds = cv_pred;
cv_results.Tests = cv_test;
end


