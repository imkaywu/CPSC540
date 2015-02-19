function [model] = adaBoost_RBFSVM(X,y,nBoosts,boostedClassifier)
    [nTrain, ~] = size(X);

    model.nBoosts = nBoosts;
    model.boostedClassifier = boostedClassifier;

    % Initialize Weights
    z = (1 / nTrain) * ones(nTrain, 1);
    alpha = zeros(nBoosts, 1);
    
    % Select part of the training set to train SVM 
    ind_1 = ones(size(X, 1), 1);
    alpha_dual = zeros(size(X, 1), 1);
    
    % RBF kernal
    sigma = 16;% Value of sigma: binary: 2, statlog: 16, sonar: 8(?), liver: 8, ionosphere: 4
    
    % error vector
    error = zeros(nBoosts, 2);% 1st column is weighted error, 2nd column is the current training error
    
    % model elements
    model.predict = @predict;

    for m = 1 : nBoosts
        % Train Weighted Classifier
        while(1)
            options.alpha_dual = alpha_dual;
            options.ind_1 = ind_1;
            options.Kernel = @rbfKernel;
            options.sigma = sigma;
            model.subModel{m} = boostedClassifier(X, y, z, options);
        
            % Compute Predictions
            yhat = model.subModel{m}.predict(model.subModel{m}, X, @rbfKernel);

            % Compute Weighted Error Rate
            err = sum(z .* (y ~= yhat));

            % Early stop
            if(err < 0.5)
                break;
            end
            sigma = sigma / 2;
        end
%         alpha_dual = model.subModel{m}.alpha;
%         ind_1 = selectTrainExp(alpha_dual, 0.9);
        
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
%     save('error/ionosphere/adaboost_svm_error.mat','error');
%     plot(error(:, 1), 'b^-');
%     hold on;
%     plot(error(:, 2), 'go-');
%     legend('weighted error', 'training error', 'b^-', 'go-');
%     axis([0 nBoosts 0 1]);
%     title('Adaboost with SVM');
end

function [y] = predict(model, X)
    alpha = model.alpha;
    y = zeros(size(X, 1), length(model.subModel));
    
    for m = 1 : length(model.subModel)
       y(:, m) = alpha(m) * model.subModel{m}.predict(model.subModel{m}, X, @rbfKernel);
    end
    y = sign(sum(y, 2));
end

function ind = selectTrainExp(alpha, pect)
    ind = zeros(size(alpha));
    total = 0;
    while(total < pect * sum(alpha))
        [alpha_val, alpha_ind] = max(alpha);
        ind(alpha_ind) = 1;
        total = total + alpha_val;
        alpha(alpha_ind) = 0;
    end
end