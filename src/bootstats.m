function[boots] = bootstats(nboots, bootprint)
% Compute the mean, standard deviation, confidence intervals and variable
% inclusion probability (VIP; frequency of non-zero values over the
% bootstrap iterations) of the bootstrapped coefficients.
% Inputs:
%   nboots: number of performed bootstrap iterations.
%   bootprint: output of the bootstrap. Specifically it must be a
%       structure array containing the bootstrapped coefficients in the 
%       Coefs field.
% Outputs:
%   boots: structure array with all the calculated measures.

boots.Mean = mean(bootprint.Coefs, 2);
boots.Median = median(bootprint.Coefs, 2);
boots.SD = std(bootprint.Coefs, 0, 2);
boots.LowerCI = boots.Mean - (1.96 * boots.SD);
boots.UpperCI = boots.Mean + (1.96 * boots.SD);
boots.VIP = (sum(bootprint.Coefs~=0, 2) * 100) / nboots;

end