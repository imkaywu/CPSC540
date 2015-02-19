clear; clc

% load data
load('iris.mat');
X = dataset(:, 1 : (size(dataset, 2) - 1));
y = dataset(:, size(dataset, 2));

% test error rate
nIter = 50;
error = zeros(nIter, 2); % the 1st column is the test error for boosting, the 2nd column is the test error for bagging
time = zeros(2, 1);

for i = 1 : nIter
    %% training and test data set
    N = size(X, 1);
    train_pect = 0.7;
    train_ind = randperm(N, train_pect * N);
    test_ind = 1 : N;
    test_ind(train_ind) = [];
    Xtrain = X(train_ind, :);
    ytrain = y(train_ind);
    Xtest = X(test_ind, :);
    ytest = y(test_ind);

    %% multi-class boosting
    % training
    tic;
    options.nBoosts = 30;
    options.classifier = 'decision tree';% or 'decision stump'
    if(strcmp(options.classifier, 'decision tree'))
        options.dt_type = 'C4.5'; % or 'ID3'
    end
    model = matLearn_classification_boosting(Xtrain, ytrain, options);

    % Testing
    yhat = model.predict(model, Xtest);
    
    % test error rate
    testError = sum(ytest ~= yhat) / size(ytest, 1);
    error(i, 1) = testError;
    fprintf('Averaged absolute test error with boosting is: %.3f\n', testError);
    time(1) = time(1) + toc;
    %% multi-class bagging
    % training
    tic;
    addpath 'multi-class bagging(Shashin Sharan''s code)';
    options.nModel = 200;
    [model_bagging] = matLearn_classification_bagging(Xtrain,ytrain,options);

    % test
    yhat = model_bagging.predict(model_bagging,Xtrain,ytrain,Xtest,options);

    % test error rate
    testError = sum(yhat~=ytest)/length(ytest);
    error(i, 2) = testError;
    fprintf('Averaged absolute test error with %s is: %.3f\n',model_bagging.name,testError);
    time(2) = time(2) + toc;
end
fprintf('Averaged training and testing time for boost and bagging are : %.3f, %.3f\n', time(1) / nIter, time(2) / nIter);
plot(error(:, 1), 'b^-');
hold on;
plot(error(:, 2), 'go-');
legend(['boosting with ', options.classifier], 'bagging', 'b^-', 'go-');
axis([1 nIter 0 1]);
title('test error of boosting and bagging');