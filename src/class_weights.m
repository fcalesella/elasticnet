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