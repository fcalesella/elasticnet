function [bootprint] = bootstrap(nboots, x, y, params)
% Perform bootstrap with an elastic net penalized fit.
% Inputs:
%   nboots: number of bootstrap iterations to be performed
%   x: n-by-p data matrix, where n is the number of observations and p the
%       number of predictors.
%   y: target measure.
%   params: structure array containing the parameters to be passed to
%       lassoglm.
% Outputs:
%   bootprint: structure array containing the results of the bootstrap.

% Initialize the variables to be saved.
[nsubj, nfeats] = size(x);
boot_coefs = zeros(nfeats + 1, nboots);
boot_alpha = zeros(1, nboots);
boot_lambda = zeros(1, nboots);

% Copy x and y to each worker.
x = parallel.pool.Constant(x);
y = parallel.pool.Constant(y);

parfor i = 1:nboots
    
    % Sample independent population.
    bootndx = randsample(nsubj,nsubj,'true');
    xb = x.Value(bootndx,:);
    yb = y.Value(bootndx,:);
    
    % Fit the model optimizing the alpha and the lambda hyper-parameters.
    [coef, best_alpha, best_lambda] = partuner(xb,yb,params);
    boot_coefs(:, i) = coef;
    boot_alpha(i) = best_alpha;
    boot_lambda(i) = best_lambda;
    
end

% Save the results in a structure array.
bootprint.Coefs = boot_coefs;
bootprint.Alpha = boot_alpha;
bootprint.Lambda = boot_lambda;

end