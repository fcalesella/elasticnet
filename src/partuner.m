function[coef, best_alpha, best_lambda] = partuner(X,y,params)
% Perform cross-validated tuning of the alpha and lambda hyper-parameters.
% Inputs:
%   X: n-by-p data matrix, where n is the number of observations and p the
%       number of predictors.
%   y: target measure.
%   params: structure array containing the parameters to be passed to
%       lassoglm.
% Outputs:
%   coef: coefficients estimated with the best hyper-parameter
%       configuration. Please note that the returned coefficients are 
%       estimated on the entire dataset (X).
%   best_alpha: value of alpha that minimizes the deviance (in combination
%       with a specific value of lambda).
%   best_lambda: value of ldambda that minimizes the deviance (in 
%       combination with a specific value of alpha).

% Assign parameters for lassoglm.
type = params.RegressionType;
k = params.KInner;
standardize = params.Standardize;
alpha = params.Alpha;
options = params.Options;

% Create train and test partitions.
c = cvpartition(y, 'KFold', k);
% Initialize the variables to be saved.
nalpha = length(alpha);
dev = zeros(nalpha, 1);
lambda = zeros(nalpha, 1);

% If class weihgts are required, then compute class weights, otherwise
% assign the same weight to the observations (lassoglm default weights).
if params.Weighted
    weights = class_weights(y);
else
    nsubj = size(X, 1);
    weights = 1/nsubj*ones(nsubj,1);
end

parfor av = 1:nalpha
    
    % For each value of alpha , fit the model optimizing the lambda 
    % hyper-parameter.
    [~,FitInfo] = lassoglm(X,y,type,'Alpha',alpha(av),'CV',c, 'Standardize', standardize,...
        'Options', options, 'Weights', weights);
    idxLambda = FitInfo.IndexMinDeviance;%modificare FitInfo.Index1SE o FitInfo.IndexMinDeviance in base alla lamba che vuoi scegliere
    lambda(av) = FitInfo.Lambda(idxLambda);
    dev(av) = FitInfo.Deviance(idxLambda);
    
end

% Find the alpha and lambda associated to the minimum deviance.
[~, min_idx] = min(dev);
best_alpha = alpha(min_idx);
best_lambda = lambda(min_idx);

% Refit the model using the best hyper-parameter combination.
[coef, FitInfo] = lassoglm (X,y,type,'Alpha',best_alpha,'Lambda', best_lambda,...
    'Standardize', standardize, 'Options', options, 'Weights', weights);
coef = [FitInfo.Intercept; coef];

end