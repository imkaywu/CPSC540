function [X_sample, y_sample] = weighted_train_sample(X, y, z, nSample)
% [X_sample, y_sample] = weighted_train_sample(X, y, z, nSample)
% 
% Sample new data based on weights
% 
% author:
% Kai Wu(12/2014)

    X_sample = zeros(nSample, size(X, 2));
    y_sample = zeros(nSample, 1);
    prob = cumsum(z);
    for i = 1 : nSample
        ind = find(rand < prob, 1);
        X_sample(i, :) = X(ind, :);
        y_sample(i) = y(ind);
    end
end