function [model] = adaBoost(X,y,nBoosts,boostedClassifier, options)
% matLearn_classification_boosting(X,y,options)
%
% Description:
%    - Adaboost algorithm
% 
% Options
%    - nBoosts: Specify the number of base learners
%    - classifier: Specify the adaboost learning strategy (one-vs-rest or SAMME) and type of base learner
%    - dt_type: Specify the type of decision tree, 'ID3' or 'C4.5'
% 
% Model
%    - nBoosts: Specify the number of base learners
%    - boostedClassifier: type of base learner
%    - subModel: sub-model
%    - alpha: voting values of each base learner computed using Adaboost algorithm
%    - label: possible values of y
%
% Author:
%  - Kai Wu (12/2014)
    [nTrain, ~] = size(X);

    model.nBoosts = nBoosts;
    model.boostedClassifier = boostedClassifier;

    % Initialize Weights
    z = (1 / nTrain) * ones(nTrain, 1);
    alpha = zeros(nBoosts, 1);
    label = unique(y);
    nClass = numel(label);

    for m = 1 : nBoosts
        % Train Weighted Classifier
        if(isequal(boostedClassifier, @decisionStump))
            model.subModel{m} = boostedClassifier(X, y, z);
        elseif(isequal(boostedClassifier, @decision_tree))
            [Xsmp, ysmp] = weighted_train_sample(X, y, z, nTrain);% refer to 'www.ccs.neu.edu/home/vip/teach/MLcourse/4_boosting/HW4/HW_boosting.html' Problem 8 Option 2
            model.subModel{m} = boostedClassifier(Xsmp, ysmp, options);
        end
        
        % Compute Predictions
        yhat = model.subModel{m}.predict(model.subModel{m}, X);

        % Compute Weighted Error Rate
        err = sum(z .* (y ~= yhat));
        
        % Compute alpha
        alpha(m) = log((1 - err) / (err + eps)) + log(nClass - 1);% without (1/2), the most important change, refer to 'Zhu J, Zou H, Rosset S, et al. Multi-class adaboost[J]. Statistics and Its, 2009.'

        % Update Weights
        z = z .* exp(alpha(m) * (1 / nClass * (y ~= yhat) - (nClass - 1) / nClass * (y == yhat)));% when nClass = 2, change back to the binary case
        
        % Re-normalize Weights
        z = z / sum(z);
    end

    model.alpha = alpha;
    model.label = label;
end