%% PARAMETERS SPECIFICATION

% Specify the seed for random processes, for reproducibility. Seed can be a
% scalar or a string (e.g., 'Default'; please refer to the MATLAB function
% rng). If 'false' is given to seed, no seed will be set.
seed = false;
%% DATA PARAMETERS

% Specify the full path of the database.
data_options.DataPath = '../demo/synthetic_data_continuous.xlsx';
% Specify the name of the spreadsheet to be used. Leave empty to lead the 
% default sheet (the first one).
data_options.Sheet = [];
% Specify the outcome variable (column number).
data_options.Target = 1;
% Specify the model indipendent variables (column number)
data_options.Predictors = (2:21);
% Specify regression type. Type: 'binomial' for logistic regression or
% 'normal' for linear regression.
data_options.TargetType = 'normal';
% Specify normalization type. Type: 'standard' for standardization or 
% 'minmax' for min-max normalization. 
data_options.Normalization = 'standard';

%% CROSS-VALIDATION PARAMETERS

% Specify the number of folds in the outer CV. 
cv_options.KOuter = 5;
% Specify the number of folds in the inner CV. 
cv_options.KInner = 5;
% Specify the values of the alpha hyper-parameter (trade-off between the L1
% and L2 regularizations). Alpha can be scalar or a numeric vector. If a
% numeric vector is provided, the alpha value will be optimized in the
% cross-validation.
cv_options.Alpha = [0.001, 0.5, 1];
% Specify the set of values on which the lambda hyper-parameter will be
% optimized. If empty, the default optimization will be performed, otherwise 
% define a sequence of values such as logspace(-5, 5, 100), in order to test
% the lambda on 100 values from 10e-5 to 10e5.
cv_options.Lambda = [];
% Specify if class weights will be assigned to the observations. This might 
% be useful in the context of classification on imbalanced data. Type:
% true to assign class weights or false otherwise.
cv_options.Weighted = false;

%% BOOTSTRAP PARAMETERS

% Specify the model that will be undergo bootstrap. Type 'optimize' to
% perform the optimization of the lambda and alpha hyper-parameters as 
% specified in the cross-validation settings. Otherwise, set a callable 
% (i.e., a function). Callables are passed by putting @ before the desired 
% function (@function). Some examples of functions that can be used are 
% central tendency mesures(e.g., @mean, @median, or @mode)
bootstrap_options.ModelDefiner = @median;
% Specify number of bootstrap iterations. 
bootstrap_options.NResamples = 500;
% Specify the method to calculate confidence intervals (CIs). Type: 'norm' for 
% normal CIs, 'per' for percentile CIs, 'cper' for corrected percentile CIs,
% 'bca' for bias-corrected CIs, or 'stud' for studentized CIs. Check the 
% MATLAB page of bootci (https://it.mathworks.com/help/stats/bootci.html) for
% further details on the options.
bootstrap_options.BootstrapType = 'norm';
% Specify the significance level.
bootstrap_options.Alpha = 0.05;
% Specify the number of resmaplings in the inner bootstrap loop for the 
% calculation of the studentized standard error (SE) estimate. This option
% will be ignored if the BootstrapType is not studentized ('stud'). The 
% MATLAB default for this parameter is 100. Check the MATLAB page of bootci 
% (https://it.mathworks.com/help/stats/bootci.html) for further details on 
% this option. 
bootstrap_options.SE = 100;

%% RUN CROSS-VALIDATION

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
fprintf('\n\nCross-validation results:\n')

% Calculate accuracy metrics and create the plots.
[measures, figures] = metricsnplots(cv_options.RegressionType, cv_results);

%% RUN BOOTSTRAP

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