function model = decisionStump(X, y, weights)
% model = decisionStump(X, y, weights)
%
% Description:
%    - decision stump

% Parameter:
%  Input:
%     X is an NxM matrix, where N is the number of points and M is the
%     number of features.
%     y is an Nx1 vector of classes
%     weights are weights for the data
%  Output:   
%    model
%    - var: the index of feature
%    - threshold: split location of the corresponding feature
%    - type: 'gt' means right/top => +1, left/bottom => -1; 'lt' means right/top => -1, left/bottom => +1
%
% Author:
%  - Kai Wu (12/2014)
    nFeatures = size(X, 2);
    
    minErr = inf;
    minVar = 0;
    minThreshold = 0;
    minThresholdType = '';
    for j = 1 : nFeatures
        thresholds = [min(X(:, j)) - eps; sort(unique(X(:, j))); max(X(:, j)) + eps];
        for t = thresholds'
            err = sum(weights(y == -1) .* (X(y == -1, j) > t)) + sum(weights(y == 1) .* (X(y == 1, j) <= t));
            
            if err < minErr
                minErr = err;
                minVar = j;
                minThreshold = t;
                minThresholdType = 'gt';
            end
            
            err = sum(weights) - err;
            if err < minErr
                minErr = err;
                minVar = j;
                minThreshold = t;
                minThresholdType = 'lt';
            end
        end
    end
    model.var = minVar;
    model.threshold = minThreshold;
    model.type = minThresholdType;
    model.predict = @predict;
end

function y = predict(model, X)
    if strcmp(model.type, 'lt')
        y = X(:, model.var) <= model.threshold;
    else
        y = X(:, model.var) > model.threshold;
    end
    y = sign(y - 0.5);
end