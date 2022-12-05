function [] = elasticnet(dataset, target, predictors, target_type, ...
    normalization_type, kouter, kinner, varargin)
% Run nested cross-validated elastic-net. Optionally perform bootstrap 
% and/or permutation test.
% Inputs: 
% 	dataset: string containing the full path to the database.
% 	target: specify the outcome variable (column number).
% 	predictors: specify the model indipendent variables (column number).
% 	target_type: specify regression type. Type: 'binomial' for logistic 
% 		regression or 'normal' for linear regression.
% 	normalization_type: specify normalization type. Type: 'standard' for 
% 		standardization or 'minmax' for min-max normalization. 
% 	kouter: specify the number of folds in the outer CV. 
% 	kinner: specify the number of folds in the inner CV.
% NameValue arguments:
% 	'Seed', false: specify the seed for random processes, for 
% 		reproducibility. Seed can be a scalar or a string (e.g., 
% 		'Default'; please refer to the MATLAB function rng). If 
% 		'false' is given to seed, no seed will be set.
% 	'Sheet', []: string specifying the name of the spreadsheet to be 
% 		used. Leave empty to load the default sheet (the first one).
% 	'Alpha', 0.5: specify the values of the alpha hyper-parameter 
% 		(trade-off between the L1 and L2 regularizations). Alpha can 
% 		be scalar or a numeric vector. If a numeric vector is provided,
% 		the alpha value will be optimized in the cross-validation.
% 	'Lambda', []: Specify the set of values on which the lambda 
% 		hyper-parameter will be optimized. If empty, the default
% 		MATLAB optimization of lassoglm 
% 		(https://it.mathworks.com/help/stats/lassoglm.html) will be 
% 		performed, otherwise define a sequence of values such as 
% 		logspace(-5, 5, 100), in order to test the lambda on 100 
% 		values from 10e-5 to 10e5.
% 	'Weighted', false: specify if class weights will be assigned to the 
% 		observations. This might be useful in the context of 
% 		classification on imbalanced data. Type: true to assign class 
% 		weights or false otherwise.
% 	'Bootstrap', false: boolean defining whether to perform the bootstrap 
% 		procedure (true) or not (false).
% 	'NResamples', 5000: specify number of bootstrap iterations. This 
% 		parameter is ignored if 'Bootstrap' is false. 
% 	'BootstrapType', 'norm': specify the method to calculate confidence 
% 		intervals (CIs). Type: 'norm' for normal CIs, 'per' for 
% 		percentile CIs, 'cper' for corrected percentile CIs, 'bca' 
% 		for bias-corrected CIs, or 'stud' for studentized CIs. Check 
% 		the MATLAB page of bootci 
% 		(https://it.mathworks.com/help/stats/bootci.html) for further 
% 		details on the options. This parameter is ignored if 
% 		'Bootstrap' is false. 
% 	'SE', 100: specify the number of resmaplings in the inner bootstrap 
% 		loop for the calculation of the studentized standard error 
% 		(SE) estimate. This option will be ignored if the BootstrapType 
% 		is not studentized ('stud'). The MATLAB default for this 
% 		parameter is 100. Check the MATLAB page of bootci 
% 		(https://it.mathworks.com/help/stats/bootci.html) for further 
% 		details on this option. This parameter is ignored if 
% 		'Bootstrap' is false. 
% 	'Permute', false: boolean defining whether to perform the 
% 		permutations (true) or not (false).
% 	'NIterations', 5000: specify number of permutation iterations. This 
% 		parameter is ignored if 'Permute' is false. 
% 	'ModelDefiner', @median: specify the model that will undergo 
% 		bootstrap or permutations. Type 'optimize' to perform the 
% 		optimization of the lambda and alpha hyper-parameters as 
% 		specified in the cross-validation settings. Otherwise, set a 
% 		callable (i.e., a function). Callables are passed by putting 
% 		@ before the desired function (@function). Some examples of 
% 		functions that can be used are central tendency mesures (e.g., 
% 		@mean, @median, or @mode). This parameter is ignored if 
% 		'Bootstrap' and 'Permute' are false. 
% Outputs:
% 	data_options: is a structure array containing the information needed 
% 		to properly handle the data.
%   cv_options: is a structure array containing the parameters passed to
%       the cross-validation procedure and the lassoglm function for 
%       elastic-net penalized regression fitting.
%   cv_results: is a structure array containing the results of the 
%           nested cross-validation procedure (the following data are saved 
%           in the structure array over the cross-validations: 
%           coefficients, best lambda value, best alpha value, AUC, ROC 
%           coordinates, true observations, and predictions).
%   performance: is a table containing the accuracy measures of the 
%     		cross-validated model.
%   figures: is a structure array containing the plots derived from the 
%     		cross-validated model.
%   bootstrap_options: is a structure array containing the parameters 
%     		passed to the bootstrap function and the information about the 
%     		model that will undergo the bootstrap procedure.
%   bootstrap_results: is a table containing the statistics derived from
%           the bootstrap procedure. The mean, median, standard deviation,
%           lower bound of the confidence intervals, upper bound of the
%           confidence intervals, and variable inclusion probability (VIP;
%           frequency of non-zero values over the bootstrap iterations) are
%           eported.
% 	X: array containing the predictors extracted from the database.
% 	y: array containing the target extracted from the database.
    
data_options.DataPath = dataset;
data_options.Target = eval(target);
data_options.Predictors = eval(predictors);
data_options.TargetType = target_type;
data_options.Normalization = normalization_type;

cv_options.KOuter = eval(kouter);
cv_options.KInner = eval(kinner);

namevalue = {'Seed', 'Sheet', 'Alpha', 'Lambda', 'Weighted',...
    'Bootstrap', 'ModelDefiner', 'NResamples', 'BootstrapType',...
    'SE', 'Permute', 'NIterations'};


for arg=1:2:length(varargin)
    if strcmp(varargin{arg}, namevalue) == false
        error('NameValue parameter %s not recognized', varargin{arg})
    end
end

if any(strcmp(varargin, 'Seed'))
    seed = eval(varargin{find(strcmp(varargin, 'Seed')) + 1});
else
    seed = false;
end

if any(strcmp(varargin, 'Sheet'))
    data_options.Sheet = varargin{find(strcmp(varargin, 'Sheet')) + 1};
else
    data_options.Sheet = [];
end

if any(strcmp(varargin, 'Alpha'))
    cv_options.Alpha = eval(varargin{find(strcmp(varargin, 'Alpha')) + 1});
else
    cv_options.Alpha = 0.5;
end

if any(strcmp(varargin, 'Lambda'))
    cv_options.Lambda = eval(varargin{find(strcmp(varargin, 'Lambda')) + 1});
else
    cv_options.Lambda = [];
end

if any(strcmp(varargin, 'Weighted'))
    cv_options.Weighted = eval(varargin{find(strcmp(varargin, 'Weighted')) + 1});
else
    cv_options.Weighted = false;
end

if any(strcmp(varargin, 'Bootstrap'))
    
    if eval(varargin{find(strcmp(varargin, 'Bootstrap')) + 1})
        
        boot = true;

        if any(strcmp(varargin, 'ModelDefiner'))
            bootstrap_options.ModelDefiner = eval(varargin{find(strcmp(varargin, 'ModelDefiner')) + 1});
        else
            bootstrap_options.ModelDefiner = @median;
        end

        if any(strcmp(varargin, 'NResamples'))
            bootstrap_options.NResamples = eval(varargin{find(strcmp(varargin, 'NResamples')) + 1});
        else
            bootstrap_options.NResamples = 5000;
        end

        if any(strcmp(varargin, 'BootstrapType'))
            bootstrap_options.BootstrapType = varargin{find(strcmp(varargin, 'BootstrapType')) + 1};
        else
            bootstrap_options.BootstrapType = 'norm';
        end

        if any(strcmp(varargin, 'SE'))
            bootstrap_options.SE = eval(varargin{find(strcmp(varargin, 'SE')) + 1});
        else
            bootstrap_options.SE = 100;
        end
    end
else
    boot = false;
end
    
if any(strcmp(varargin, 'Permute'))
    
    if eval(varargin{find(strcmp(varargin, 'Permute')) + 1})
        
        permute = true;
        
        if any(strcmp(varargin, 'ModelDefiner'))
            permutation_options.ModelDefiner = eval(varargin{find(strcmp(varargin, 'ModelDefiner')) + 1});
        else
            permutation_options.ModelDefiner = @median;
        end

        if any(strcmp(varargin, 'NIterations'))
            permutation_options.NIterations = eval(varargin{find(strcmp(varargin, 'NIterations')) + 1});
        else
            permutation_options.NIterations = 5000;
        end
    end
    
else
    permute = false;
end

% If parallel computing has not been started yet, initialize it.
try
    parpool;
catch
end

% If requested set the seed for random processes.
if seed
    spmd %#ok<UNRCH>
        rng(seed);
    end
    rng(seed);
end

% Load and prepare the data.
[X, y, ~] = data_loader(data_options);

fprintf(['\nRunning cross-validation...\n'...
    'Selected parameters:\n'...
    'External CV: %i fold\n'...
    'Internal CV: %i fold'], cv_options.KOuter, cv_options.KInner)

% Share information about TargetType with cv_options
cv_options.RegressionType = data_options.TargetType;

% Perform nested cross-validation.
cv_results = outer_cv(X, y, cv_options);

fprintf('\n\nCross-validation state: done.')

% Calculate accuracy metrics and create the plots.
[performance, figures] = metricsnplots(cv_options.RegressionType, cv_results);

fprintf('\n\nCross-validation results:\n')
disp(performance)

if boot

    % If requested set the seed for random processes.
    if seed
        spmd %#ok<UNRCH>
            rng(seed);
        end
        rng(seed);
    end

    % Load and prepare the data.
    [X, y, bootstrap_options.Names] = data_loader(data_options);

    % Select the proper model for bootstrap.
    if strcmp(bootstrap_options.ModelDefiner, 'optimize')
        bootstrap_options.Model = cv_options;
    elseif isa(bootstrap_options.ModelDefiner, 'function_handle')
        bootstrap_options.Model = cv_options;
        bootstrap_options.Model.Alpha = bootstrap_options.ModelDefiner(cv_results.Alpha);
        bootstrap_options.Model.Lambda = bootstrap_options.ModelDefiner(cv_results.Lambda);
    else
        error('Type of model for bootstrap not supported')
    end

    fprintf(['\nRunning Bootstrap...\n'...
        'Selected number of iterations: %i'], bootstrap_options.NResamples);

    % Perform bootstrap.
    bootstrap_results = bootstrap(X, y, bootstrap_options);

    fprintf('\n\nBootstrap state: done.')
    fprintf('\n\nBootstrap statistics:\n')
    disp(bootstrap_results)
end

if permute
    
    % If requested set the seed for random processes.
    if seed
        spmd %#ok<UNRCH>
            rng(seed);
        end
        rng(seed);
    end

    % Load and prepare the data.
    [X, y, permutation_options.Names] = data_loader(data_options);

    % Select the proper model for bootstrap.
    if strcmp(permutation_options.ModelDefiner, 'optimize')
        permutation_options.Model = cv_options;
    elseif isa(permutation_options.ModelDefiner, 'function_handle')
        permutation_options.Model = cv_options;
        permutation_options.Model.Alpha = permutation_options.ModelDefiner(cv_results.Alpha);
        permutation_options.Model.Lambda = permutation_options.ModelDefiner(cv_results.Lambda);
    else
        error('Type of model for permutations not supported')
    end

    fprintf(['\nRunning Permutations...\n'...
        'Selected number of iterations: %i'], permutation_options.NIterations);

    % Perform bootstrap.
    [pval, null_stat] = permute_model(X, y, permutation_options.Model,...
        permutation_options.NIterations);

    fprintf('\n\nPermutations state: done.')
    fprintf('\n\nPermutations p-value: %d', pval)
end

clear ans boot dataset kinner kouter normalization_type permute...
    predictors seed target target_type varargin
save('elasticnet_results.mat');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [bootstrap_results] = bootstrap(X, y, bootstrap_options)
% Perform bootstrap of the partuner funciton.
% Inputs:
%   X: n-by-p data matrix, where n is the number of observations and p the
%       number of predictors.
%   y: target measure.
%   boostrap_options: structure array containing the parameters to be 
%       passed to bootci.
% Outputs:
%   bootstrap_results: structure array with bootstrap statics.

% Define the funciton to be bootstrapped that will be partuner.
bootfun = @(x, y, options)partuner(x, y, options);

% Run the bootstrap and compute confidence intervals.
[ci, bootstat] = bootci(bootstrap_options.NResamples,...
        {bootfun, X, y, bootstrap_options.Model},...
        'Type', bootstrap_options.BootstrapType,...
        'NBootStd', bootstrap_options.SE,...
        'Options', statset('UseParallel', true));

% Compute bootstrap statistics.
boots.Variables = ['Intercept'; bootstrap_options.Names];
boots.Mean = mean(bootstat, 1)';
boots.Median = median(bootstat, 1)';
boots.SD = std(bootstat, 0, 1)';
boots.LowerCI = ci(1, :)';
boots.UpperCI = ci(2, :)';
boots.VIP = ((sum(bootstat~=0, 1) * 100) / bootstrap_options.NResamples)';
bootstrap_results = struct2table(boots);  

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[weights] = class_weights(y)
% Compute weights for classes. The function supports multi-target
% weightening, string labels, and numeric labels other than 0 and 1.
% Inputs:
%   y: numeric or string vector containing the class to which each
%       observation belongs (i.e., target vector).
% Outputs:
%   weights: numeric vector containing the weight associated to the class
%       of each observation.

% Find the number of classes.
classes = unique(y);
nclass = length(classes);
% Initialize the weights to 0.
nsample = length(y);
weights = zeros(1, nsample);

for cl = 1:nclass
    
    % Find observations belonging to a specific class, and assign the
    % corresponding weight.
    posclass = y==classes(cl);
    weights(posclass) = 1 / (nclass * sum(posclass));
    
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[x, y, names] = data_loader(data_options)
% Perform data preparation and sanity check. 
% Inputs:
%   data_options: a sructure array with the following fields:
%       data_path: full path to the databse
%       sheet: name of the spreadsheet to be leaded (if any, else set to
%       false)
%       predictors: indeces of the columns to select for the predictors
%       target: index of the column to select for the target
%       normalization: normalization type ("standard" or "minmax")
%       regression_type: indicates the type of target ("normal" for
%       continuous targets or "binomial" for dichotomous targets)
% Outputs:
%   x: array containing the predictor data.
%   y: array containing the target data.
%   names: array containing the names of the selected predictors.

% Load the dataset.
warning('off')
if data_options.Sheet
    data = readtable(data_options.DataPath, 'Sheet', data_options.Sheet);
else
    data = readtable(data_options.DataPath);
end
warning('on')

% Get variable names.
names = data.Properties.VariableNames';
% Convert table to array.
data = table2array(data);
% Only the names of the selected variables will be needed.
names = names(data_options.Predictors); 

% Sanity check: if NaN values are present, then raise a warning.
if any(isnan(data(:)))
    warning('Data contains missing values. This could cause errors and/or data neglect');
end

% Create the predictor and target variables.
x = data(:, data_options.Predictors);
y = data(:, data_options.Target);

% Remove cases with missing values (NaNs).
idx_nan = any(isnan(x), 2);
x(idx_nan, :) = [];
y(idx_nan) = [];

% Raise warning stating how many cases will be discarded.
if sum(idx_nan) > 0
    warning('%i casewise removals occurred because of missing values', ...
        sum(idx_nan));
end

% Normalize the data.
if strcmp(data_options.Normalization, 'standard')
    x = (x - mean(x, 1)) ./ std(x, 1);
    if strcmp(data_options.TargetType, 'normal')
        y = (y - mean(y)) ./ std(y);
    elseif strcmp(data_options.TargetType, 'binomial')
        % do nothing
    else
        error('Type of target "%s" does not exist', data_options.TargetType)
    end
elseif strcmp(data_options.Normalization, 'minmax')
    x = (x - min(x, [], 1)) ./ (max(x, [], 1) - min(x, [], 1));
    if strcmp(data_options.TargetType, 'normal')
        y = (y - min(y, [], 1)) ./ (max(y, [], 1) - min(y, [], 1));
    elseif strcmp(data_options.TargetType, 'binomial')
        % do nothing
    else
        error('Type of target "%s" does not exist', data_options.TargetType)
    end
else
    error('Type of normalization "%s" does not exist', data_options.Normalization)
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [measures, figures] = metricsnplots(reg_type, cv)
% Compute accuracy measures and create the plots.
% Inputs:
%   reg_type: type of regression. If logistic regression has been performed
%       pass 'binomial', otherwise 'normal' if linear regression was
%       performed.
%   cv: structure array containing the test (field Tests) and predicted 
%       (field Preds) observations. If logistic regression was performed 
%       and the AUC field is present, than it assumed that the structure 
%       array contains the results of the cross-validation. If the AUC 
%       field is not present, than it is assumed that the structure array 
%       contains the results of the final prediction. In this last case, a
%       field with the continous predictions (field PredsContinuous,
%       containing the non-binarized predictions) is needed.
% Outputs:
%   measures: structure array containing all the accuracy measures
%       computed.
%   figures: structure array containing all the plots created.

if strcmp(reg_type, 'binomial')
    
    % For classification.
    
    % Compute confusion matrix and the derived measures (true poitives, 
    % true negatives, false negatives, and false positives).
    [~,cm,~,~] = confusion(cv.Tests, cv.Preds);
    TP = cm(2,2);
    TN = cm(1,1);
    FN = cm(2,1);
    FP = cm(1,2);

    % Model sensitivity.
    Sensitivity = TP/(TP + FN);
    measures.Sensitivity = Sensitivity;

    % Model specificity. 
    Specificity = TN/(TN+FP);
    measures.Specificity = Specificity;

    % Model PPV.
    PPV = TP/(TP+FP);
    measures.PPV = PPV;
    % Model NPV.
    NPV = TN/(TN+FN);
    measures.NPV = NPV;

    % Balance Accuracy.
    BalanceAccuracy = (Sensitivity + Specificity)/2;
    measures.BalanceAccuracy = BalanceAccuracy;

    % Diagnostic Odd Ratio.
    dorn = Specificity*Sensitivity;
    dord = (1 - Specificity) * (1 - Sensitivity);
    DOR = dorn/dord;
    measures.DOR = DOR;

    % Plot confusion matrix for accuracy.
    f1 = figure;
    plotconfusion(cv.Tests, cv.Preds)
    figures.f1 = f1;
    
    if isfield(cv, 'AUC')
        
        k = length(cv.AUC);
        auc_mean = mean(cv.AUC); 
        auc_sd = std(cv.AUC);
        measures.AUCMean = auc_mean;
        measures.AUCSD = auc_sd;
        
        % Mean of folds (coordinate of ROC curve).
        XmeanROC = foldmean (cv.XFold);
        YmeanROC = foldmean (cv.YFold);
        % Plot ROC.
        f2 = figure;
        legend_var = plot_roc(cv.XFold, cv.YFold, k);
        plot(XmeanROC,YmeanROC, 'r ', 'LineWidth', 2)
        xlabel('False positive rate'); 
        ylabel('True positive rate');
        title_var = sprintf('ROC for %i folds Cross Validation', k);
        title(title_var)
        legend(legend_var, 'Mean across Folds')
        hold off
        figures.f2 = f2;
        
    else
        
        % If the AUC field is not present in cv, calculate it and plot the
        % ROC.
        f2= figure;
        [xfold,yfold,~,auc] = perfcurve(cv.Tests,cv.PredsContinuous,'1');
        plot(xfold,yfold,'r ', 'LineWidth', 2);
        xlabel('False positive rate'); ylabel('True positive rate');
        title('ROC Curves of bootstrap coefficients')
        fprintf('AUC = %.4f\n', auc);
        measures.AUC = auc;
        figures.f2 = f2;
        
    end
    
else
    
    % For linear regression.
    
    % Model accuracy with MSE.
    MSE = mean((cv.Preds - cv.Tests).^2);
    measures.MSE = MSE;

    % Model accuracy with RMSE.
    RMSE = sqrt(MSE);
    measures.RMSE = RMSE;

    % Model accuracy with R square.
    sst = sum((cv.Tests - mean(cv.Tests)).^2);
    sse = sum((cv.Preds - cv.Tests).^2);
    Rsq = 1 - sse / sst;
    measures.Rsq = Rsq;
    
    % Plot predictions.
    f1 = figure();
    f1(1) = scatter(cv.Tests, cv.Preds, 'filled', 'MarkerFaceColor', 'k');
    f1(2) = lsline;
    f1(2).Color = 'k';
    xlabel('True value')
    ylabel('Prediciton')
    figures.f1 = f1;
    
end

measures = struct2table(measures);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[coef, best_alpha, best_lambda] = partuner(X,y,cv_options)
% Perform cross-validated tuning of the alpha and lambda hyper-parameters.
% Inputs:
%   X: n-by-p data matrix, where n is the number of observations and p the
%       number of predictors.
%   y: target measure.
%   cv_options: structure array containing the parameters to be passed to
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
type = cv_options.RegressionType;
k = cv_options.KInner;
alpha = cv_options.Alpha;
lambda = cv_options.Lambda;
options = statset('UseParallel', true);

% Create train and test partitions.
if strcmp(type, 'normal')
    c = cvpartition(length(y), 'KFold', k);
else
    c = cvpartition(y,'KFold', k);
end

% Initialize the variables to be saved.
nalpha = length(alpha);
dev = zeros(nalpha, 1);
lambda_for_alpha = zeros(nalpha, 1);

% If class weihgts are required, then compute class weights, otherwise
% assign the same weight to the observations (lassoglm default weights).
if strcmp(type, 'normal')
    nsubj = size(X, 1);
    weights = 1/nsubj*ones(nsubj,1);
else
    if cv_options.Weighted
        weights = class_weights(y);
    else
        nsubj = size(X, 1);
        weights = 1/nsubj*ones(nsubj,1);
    end
end

parfor av = 1:nalpha
    
    % For each value of alpha , fit the model optimizing the lambda 
    % hyper-parameter.
    if lambda
        [~,FitInfo] = lassoglm(X,y,type,'Alpha',alpha(av),'CV',c,...
            'Lambda', lambda, 'Options', options, 'Weights', weights);
    else
        [~,FitInfo] = lassoglm(X,y,type,'Alpha',alpha(av),'CV',c,...
            'Options', options, 'Weights', weights);
    end
    
    idxLambda = FitInfo.IndexMinDeviance;%modificare FitInfo.Index1SE o FitInfo.IndexMinDeviance in base alla lamba che vuoi scegliere
    lambda_for_alpha(av) = FitInfo.Lambda(idxLambda);
    dev(av) = FitInfo.Deviance(idxLambda);
end

% Find the alpha and lambda associated to the minimum deviance.
[~, min_idx] = min(dev);
best_alpha = alpha(min_idx);
best_lambda = lambda_for_alpha(min_idx);

% Refit the model using the best hyper-parameter combination.
[coef, FitInfo] = lassoglm (X,y,type,'Alpha',best_alpha,...
    'Lambda', best_lambda, 'Options', options, 'Weights', weights);
coef = [FitInfo.Intercept; coef];

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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