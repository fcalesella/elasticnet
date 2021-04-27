%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% PARAMETER SPECIFICATION %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Specify the database path and name. If there are multiple sheets use 
% ('Database_name.xlsx', 'Sheet', 'sheet_name').
data = readtable('synthetic_data.xlsx'); 
% Specify the outcome variable (column number).
y_sel = (1);
% Specify the model indipendent variables (column number) for
% cross-validation.
cv_var_sel = (2:21);
% Specify the model indipendent variables (column number) for bootstrap.
boot_var_sel = (2:21);
% Specify regression type. Select: 'binomial' for logistic regression;
% 'normal' for linear regression.
reg_type = 'binomial';
% Specify the number of folds in the outer CV. 
k = 10;
% Specify the number of folds in the inner CV. 
k_int = 10;
% Specify the value of the alpha hyper-parameter (trade-off between the L1
% and L2 regularizations). Alpha can be scalar or a numeric vector. If a
% numeric vector is provided, the alpha value will be optimized in the
% cross-validation.
cv_alpha = 0.5;
% Specify if lassoglm will estimate coefficients on standardized data.
% Select: 'true' for coefficient estimation on standardized data; 'false'
% otherwise. Please note that the output coefficients will be in the raw
% space anyway.
stand = true;
% Specify if class weights will be assigned the observations. This might be
% useful in the context of classification on imbalanced data. Select:
% 'true' to assign class weights; 'false' otherwise.
weighted = true;
% Specify if lassoglm will use parallel computing. Select: 'true' to use
% the Parallel Computing Toolbox; 'false' otherwise. Please note that this
% only refers to the lassoglm function, not to the costumized functions.
par = true;
% Specify the alpha values to pass and optimize in the bootstrap.
% This variable can be numeric (both scalar and vector), to perform 
% parameter optimization in the boostrap procedure; or it can be a
% function. If a function is passed, the specified function will be applied
% to the vector of alpha values retuned by the cross-validation, in order
% to extract a single optimal alpha value (functions are passed by putting 
% @ before the desired function: @function). Some example functions for the
% selction can be some central tendency measures (e.g., @mean, @median, 
% and/or @mode).
boot_alpha = @median;
% Specify number of bootstrap iterations. 
nboots = 500;
% Specify if bootstrap will be stratified. This might be useful in the
% context of classification on imbalanced data. Select: 'true' to run
% stratified bootstrap; 'false' otherwise.
strat_boot = true;
% Specify the seed for random processes, for reproducibility. Seed can be a
% scalar or a string (e.g., 'Default'; please refer to the MATLAB function
% rng). If 'false' is given to seed, no seed will be set.
seed = false;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Save input parameters in the structure array params.
params.RegressionType = reg_type;
params.KOuter = k;
params.KInner = k_int;
params.CVAlpha = cv_alpha;
params.BootAlpha = boot_alpha;
params.Weighted = weighted;
params.Standardize = stand;
params.Options = statset('UseParallel', par);

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

% Extract the X and Y from the data table, for the cross-validation.
[X, y, ~] = data_manager(data, cv_var_sel, y_sel);

fprintf(['\nSelected parameters:\nExternal CV: %i fold\n'...
    'Internal CV: %i fold\nBootstrap: %i iterations\n\n'...
    'Running cross-validation...\n'], k, k_int, nboots)

% Select the proper sequence of alpha values to be tested in the
% cross-validation.
params.Alpha = params.CVAlpha;
% Perform nested cross-validation.
cv = outer_cv(X, y, params); 

fprintf('Cross-validation state: done.\n\n')

% Calculate accuracy metrics and create the plots.
[measures, figures] = metricsnplots(reg_type, cv);
% Prepare the data for the bootstrap.
[X, y, names] = data_manager(data, boot_var_sel, y_sel);
  
fprintf('\nRunning Bootstrap...\n');

% Select the proper alpha values to be tested in th einner cross-validation
% of the bootstrap.
if isa(boot_alpha,'function_handle')
    params.Alpha = params.BootAlpha(cv.Alpha);
else
    params.Alpha = params.BootAlpha;
end

% Reset the seed for the bootstrap.
if seed
    spmd %#ok<UNRCH>
        rng(seed);
    end
    rng(seed);
end

% Perform the bootstrap or its stratified version if requested.
if strat_boot
    bootprint = stratified_bootstrap(nboots, X, y, params);
else
    bootprint = bootstrap(nboots, X, y, params);%#ok<UNRCH>
end

fprintf('Bootstrap state: done.\n\nBootstrap statistics:\n\n')

% Calculate bootstrap statistics and create a table with the results.
boots = bootstats(nboots, bootprint);
boots.Variables = ['Intercept'; names];
boots = orderfields(boots, [7, 1, 2, 3, 4, 5, 6]);
resboot = struct2table(boots);  
disp(resboot)
