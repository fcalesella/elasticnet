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