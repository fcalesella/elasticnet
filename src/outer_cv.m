
function [cv] = outer_cv(X, y, params)
% Perform nested cross-validation (outer loop).
% Inputs:
%   X: n-by-p data matrix, where n is the number of observations and p the
%       number of predictors.
%   y: target measure.
%   params: structure array containing the parameters to be passed to
%       lassoglm.
% Outputs:
%   cv: structure array containing the nested cross-validation results.

% Create train and test partitions.
c = cvpartition(y,'KFold', params.KOuter);
% Initialize the variables to be saved.
cv_coef = zeros(size(X, 2) + 1, params.KOuter);
cv_lambda = zeros(1, params.KOuter);
cv_alpha = zeros(1, params.KOuter);
cv_auc = zeros(1, params.KOuter);
cv_pred = [];
cv_test = [];
% cell array with every xfold (i.e. coordinate of curve ROC)
xfoldc = cell(1, params.KOuter);
% cell array with every yfold (i.e. coordinate of curve ROC)
yfoldc = cell(1, params.KOuter);

% Copy X and y to each worker.
X = parallel.pool.Constant(X);
y = parallel.pool.Constant(y);

parfor i = 1:params.KOuter

    % Select train and test sets.
    idxTrain = training(c,i);
    idxTest = test(c, i);
    XTrain = X.Value(idxTrain,:);
    yTrain = y.Value(idxTrain);
    XTest = X.Value(idxTest,:);
    yTest = y.Value(idxTest);

    % Fit the model optimizing the alpha and the lambda hyper-parameters.
    [coef, best_alpha, best_lambda] = partuner(XTrain,yTrain,params)

    cv_coef(:, i) = coef;
    cv_lambda(i) = best_lambda;
    cv_alpha(i) = best_alpha;

    if strcmp(params.RegressionType, 'binomial')
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
cv.Coef = cv_coef;
cv.Lambda = cv_lambda;
cv.Alpha = cv_alpha;
cv.AUC = cv_auc;
cv.XFold = xfoldc;
cv.YFold = yfoldc;
cv.Preds = cv_pred;
cv.Tests = cv_test;
end


