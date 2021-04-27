function[x, y, names] = data_manager(data, x_sel, y_sel)
% Perform data preparation and sanity check. 
% Inputs:
%   data: table object containing the database.
%   x_sel: indeces of the predictor variables.
%   y_sel: indeces of the target variable.
% Outputs:
%   x: array containing the predictor data.
%   y: array containing the target data.
%   names: array containing the names of the selected predictors.

% Get variable names.
names = data.Properties.VariableNames';
% Convert table to array.
data = table2array(data);
% Only the names of the selected variables will be needed.
names = names(x_sel); 

% Sanity check: if NaN values are present, then raise a warning.
if any(isnan(data(:)))
    warning('Data contains missing values. This could cause errors and/or data neglect');
end

% Create the predictor and target variables.
x = data(:, x_sel);
y = data(:, y_sel);

% Remove cases with missing values (NaNs).
idx_nan = any(isnan(x), 2);
x(idx_nan, :) = [];
y(idx_nan) = [];

% Raise warning stating how many cases will be discarded.
if sum(idx_nan) > 0
    warning('%i casewise removals occurred because of missing values', ...
        sum(idx_nan));
end 

end