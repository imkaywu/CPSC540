function [model] = adaBoost(X,y,nBoosts,boostedClassifier)
    [nTrain, ~] = size(X);

    model.nBoosts = nBoosts;
    model.boostedClassifier = boostedClassifier;

    % Initialize Weights
    z = (1 / nTrain) * ones(nTrain, 1);
    alpha = zeros(nBoosts, 1);
    
    % error vector
    error = zeros(nBoosts, 2);% 1st column is weighted error, 2nd column is the current training error
    
    % model elements
    model.predict = @predict;

    for m = 1 : nBoosts
        % Train Weighted Classifier
        if(isequal(boostedClassifier, @decision_tree))
            [Xsmp, ysmp] = weighted_train_sample(X, y, z, nTrain);
            model.subModel{m} = boostedClassifier(Xsmp, ysmp);
        elseif(isequal(boostedClassifier, @decision_stump) || isequal(boostedClassifier, @linear_regression))
            model.subModel{m} = boostedClassifier(X, y, z);
        end
        
        % Compute Predictions
        yhat = model.subModel{m}.predict(model.subModel{m}, X);
        
        % Compute Weighted Error Rate
        err = sum(z .* (y ~= yhat));
        
        % Early stop
        if(err >= 0.5)
            break;
        end
        
        % Compute alpha
        alpha(m) = (1 / 2) * log((1 - err) / (err + eps));

        % Early stop
        if(err == 0)
            break;
        end
        
        % Update Weights
        z = z .* exp(-alpha(m) * y .* yhat);

        % Re-normalize Weights
        z = z / sum(z);
        
        % error
        error(m, 1) = err;
        model.alpha = alpha;
        yhat = model.predict(model, X);
        error(m, 2) =sum(yhat ~= y) / numel(y);
    end
    %% comment the following codes for Experiment 2, uncomment them for Experiment 1
%     save('error/ionosphere/adaboost_dt_error.mat','error');
%     plot(error(:, 1), 'b^-');
%     hold on;
%     plot(error(:, 2), 'go-');
%     legend('weighted error', 'training error', 'b^-', 'go-');
%     axis([0 nBoosts 0 1]);
%     title('Adaboost with decision tree');
end

function [y] = predict(model, X)
    alpha = model.alpha;
    y = zeros(size(X, 1), length(model.subModel));
    
    for m = 1 : length(model.subModel)
       y(:, m) = alpha(m) * model.subModel{m}.predict(model.subModel{m}, X);
    end
    y = sign(sum(y, 2));
end