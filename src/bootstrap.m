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