function model = decision_stump(X, y, weights)
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