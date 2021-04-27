%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% PARAMETER SPECIFICATION %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Specify the path and name of the model on which to perform final
% predicitons after bootstrap.
model = 'model.mat';
% Specify the outcome variable (column number).
y_sel= (1);
% Specify the model indipendent variables (column number).
var_sel = (2:21);
% Specify if any of the variable will be discarded from the model. The
% coefficient value il be set to 0 at these positions. Select: a scalar or
% numeric vector representing the column numbers of the variables to be
% discarded; 'false' if no variable has to be discarded.
discard = false;
% Specify the central tendency measure of the coefficinets to be used. Only
% 'Mean' and 'Median' are implemented.
measure = 'Mean';
% Specify the criterion to select the included variables. Select: 'All' if
% all the variables (except for the discarded ones) are to be included;
% 'VIP' to select coefficients based on the VIP values; 'CI' to select
% variables based on the confidence intervals (only variables that do not
% include 0 in the confidence interval will be kept).
criterion = 'VIP';
% Specify the VIP threshold. Only variables with VIP>threshold will be
% kept. Please note that if the selected criterion is 'All' or 'CI', this
% parameter will be ignored.
threshold = 85;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load needed informations from the speicfied model.
load(model, 'data', 'params', 'resboot');
% Select the coefficients for the final prediciton.
final_coefs = select_coefs(resboot, discard, measure, criterion, threshold);
% Extract the X and Y for the final prediction.
[X, y, names] = data_manager(data, var_sel, y_sel);

% Retrieve regression type and set the appropriate link function.
if strcmp(params.RegressionType, 'binomial')
    link = 'logit';
else
    link = 'identity';
end

% Perform the final prediction using the selected coefficients.
y_final_Pred_con = glmval(final_coefs,X,link,'constant','on');
% Save the test and predicted observations. For classification binarize the
% predictions.
out.Tests = y.';
if strcmp(params.RegressionType, 'binomial')
    y_final_Pred = (double(y_final_Pred_con>=0.5).');
    out.Preds = y_final_Pred;
    out.PredsContinuous = y_final_Pred_con;
else
    out.Preds = y_final_Pred;
end

% Compute metrics and create plots.
[final_measures, final_figures] = metricsnplots(params.RegressionType, out);
