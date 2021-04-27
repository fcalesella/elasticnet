function[coefs] = select_coefs(resboot, discard, measure, criterion, threshold)
% Perform coefficient selection for the final prediciton. 
% Inputs:
%   resboot: table with the results of the bootstrap.
%   discard: scalar or numeric vector indexing the variables that are to be
%       excluded from the model.
%   measure: the measure of central tendency of the coefficients to be
%       used. Only 'Mean' and 'Median' are implemented.
%   criterion: criterion to select the included variables. Implemented 
%       cases are: 'All' if all the variables (except for the discarded 
%       ones) are to be included; 'VIP' to select coefficients based on the
%       VIP values; 'CI' to select variables based on the confidence 
%       intervals (only variables that do not include 0 in the confidence 
%       interval will be kept).
%   threshold: VIP threshold. Only variables with VIP>threshold will be
%       kept. Please note that if the selected criterion is 'All' or 'CI', 
%       this parameter will be ignored.
% Outputs:
%   coefs: the selcted coefficients (the non-selcted ones are set to 0).

% Get the coefficients values indicated by measure.
coefs = resboot{:, measure};

% Set coefficient values to 0 at the indeces indicated by discard.
if discard
    coefs(discard) = 0;
end

% Set to 0 the coefficients that do not respect the criterion and/or the
% threshold.
switch criterion
    
    case 'All'
        
        % Do nothing, no coefficient must be set to 0 in this case.
    
    case 'VIP'
        
        % Get the VIP values.
        vip = resboot{:, 'VIP'};
        % Get the index of the VIP values greater than or equal to the
        % threshold. Indeces are boolean, with 1 where the VIP is above 
        % threshold, and 0 where the VIP is below threshold, so just 
        % multiply the coefficients by the indeces to set to 0 the 
        % coefficients below threshold.
        idx = vip >= threshold;
        coefs = coefs .* idx;
    
    case 'CI'
        
        % Get the lower and upper CIs.
        low_ci = resboot{:, 'LowerCI'};
        high_ci = resboot{:, 'UpperCI'};
        % Get the index of the CIs that do not include 0. Indeces are 
        % boolean, with 1 where the CIs do not include the 0, and 0 where 
        % the CIs include the 0, so just multiply the coefficients by the 
        % indeces to set to 0 the coefficients with 0 included in the CIs.
        idx = ~(low_ci<0 & high_ci>0);
        coefs = coefs .* idx;
    
    otherwise
        
        % Sanity check: if a non-implemented criterion is passed, then
        % raise and error.
        error(['Error: criterion not found. The implemented criteria ',...
            'are: "All", "VIP", and "CI".'])

end

end