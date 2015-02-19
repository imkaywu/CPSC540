function model = matLearn_classification_boosting(X, y, options)
% matLearn_classification_boosting(X,y,options)
%
% Description:
%    - Multi-class classification boosting based on decision stump or
%    decision tree
% 
% Options
%    - nBoosts: Specify the number of base learners
%    - classifier: Specify the adaboost learning strategy (one-vs-rest or SAMME) and type of base learner
%    - dt_type: Specify the type of decision tree, 'ID3' or 'C4.5'
%
% Model
%  Strategy:
%    - for one-vs-rest strategy, model.subModel{i} is the one set of boosted decision
%    stumps. model.subModel{i}.subModel{j} is one decision stump
%    - for SAMME, model is the boosted decisoin trees, model.subModel{i} is one decision tree.
%  Parameters:
%    - nBoosts: Specify the number of base learners
%    - boostedClassifier: base learner
%    - subModel: sub-model
%    - alpha: voting values of each base learner computed using Adaboost algorithm
%    - label: possible values of y
%
% Author:
%  - Kai Wu (12/2014)
    nBoosts = options.nBoosts;
    classifier = options.classifier;
    if(strcmp(classifier, 'decision stump'))
        nClass = unique(y);
        model.subModel = cell(numel(nClass), 1);
        for c = nClass'% one-versus-rest boosting
            yTrain = y;
            yTrain(y ~= c) = -1;
            yTrain(y == c) = 1;
            model.subModel{c} = adaBoost(X, yTrain, nBoosts, @decisionStump);
        end
    elseif(strcmp(classifier, 'decision tree'))% multi-class boosting, refer to 'Zhu J, Zou H, Rosset S, et al. Multi-class adaboost[J]. Statistics and Its, 2009.'
        model = adaBoost(X, y, nBoosts, @decision_tree, options);
    end
    
    model.classifier = classifier;
    model.predict = @predict;
end

function y = predict(model, X)
    if(strcmp(model.classifier, 'decision stump'))
        % model here are nClass boosted decision stumps, model.subModel{i} here is one
        % boosted decision stumps, the model.subModel{i}.subModel{j} is one decision stump
        nModel = length(model.subModel);
        y = zeros(size(X, 1), nModel);
        for c = 1 : nModel
            y(:, c) = predict_binary(model.subModel{c}, X);
        end
        [~, y] = max(y, [], 2);
    elseif(strcmp(model.classifier, 'decision tree'))
        % model here is one boosted decision tree, model.subModel{i} is one decision tree.
        y = predict_multi(model, X);
    end
end

% model here is one boosted decision stump, and model.subModel{i} is a decision stump
function y = predict_binary(model, X)
    alpha = model.alpha;
    y = zeros(size(X, 1), length(model.subModel));
    
    for m = 1 : length(model.subModel)
       y(:, m) = alpha(m) * model.subModel{m}.predict(model.subModel{m}, X);
    end
    % what should I use?
%     y = sign(sum(y, 2));
    y = mean(y, 2);
end

% model here is one boosted decision trees, and model.subModel{i} is one decision tree.
function y = predict_multi(model, X)
    alpha = model.alpha;
    alpha = alpha(1 : length(model.subModel));
    label = model.label;
    y = zeros(size(X, 1), length(model.subModel));
    class = zeros(size(X, 1), numel(label));
    
    for m = 1 : length(model.subModel)
        y(:, m) = model.subModel{m}.predict(model.subModel{m}, X);
    end
    
    for c = 1 : numel(label)
        class(:, c) = sum(repmat(alpha', [size(X, 1), 1]) .* (y == label(c)), 2);
    end
    
    [~, ind] = max(class, [], 2);
    y = label(ind);
end