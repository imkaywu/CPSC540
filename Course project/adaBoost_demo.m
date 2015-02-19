clear; clc;
%% dataset 1: binary
% load('binary.mat');
% [X,~,~] = standardizeCols(X);

%% dataset 2: statlog
load('statlog.mat');
X = dataset(:, 1 : end - 1);
y = dataset(:, end);

%% dataset 3: sonar
% load('sonar.mat');
% X = dataset(:, 1 : end - 1);
% X = standardizeCols(X);
% y = dataset(:, end);

%% dataset 4: liver
% load('liver.mat');

%% dataset 5; ionosphere dataset
% load('ionosphere.mat');
% X = dataset(:, 1 : end - 1);
% X = standardizeCols(X);
% y = dataset(:, end);

%% dataset 6; credit(https://archive.ics.uci.edu/ml/datasets/Credit+Approval, missing data)
% load('credit.mat');
% X = dataset(:, 1 : end - 1);
% X = standardizeCols(X);
% y = dataset(:, end);
% y(y == 0) = -1;

%% dataset 7; diabetes
% load('diabetes.mat');
% X = dataset(:, 1 : end - 1);
% X = standardizeCols(X);
% y = dataset(:, end);

%% dataset8: australian
% load('australian.mat');
% [X,~,~] = standardizeCols(X);
% N = size(X,1);
% ind = randperm(N, N / 2);
% index = 1 : N;
% index(ind) = [];
% Xtest = X(index, :);
% ytest = y(index, :);
% X = X(ind, :);
% y = y(ind, :);

%% Experiment 1
% % Train
% nBoosts = 30;
% 
% % Fit Boosted model
% addpath 'Base Learner';
% addpath 'Kernel'
% % model = adaBoost(X, y, nBoosts, @decision_tree);% or decision_stump
% model = adaBoost_RBFSVM(X, y, nBoosts, @SVM_Kernel);
% 
% % Compute training error
% yhat = model.predict(model,X);
% trainError = sum(yhat ~= y) / numel(y);
% 
% % Show data and decision boundaries
% % plot2DClassifier(X,y,model);

%% Experiment 2
nIter = 50;
nBoosts = 10;
error = zeros(nIter, 2); %1st column: train error, 2nd column: test error.
N = size(X, 1);
for i = 1 : nIter
    ind_train = randperm(N, round(0.5 * N));
    ind_test = 1 : N;
    ind_test(ind_train) = [];
    Xtrain = X(ind_train, :);
    ytrain = y(ind_train);
    Xtest = X(ind_test, :);
    ytest = y(ind_test);
    % Train

    % Fit Boosted model
    addpath 'Base Learner';
    addpath 'Kernel';
    model = adaBoost(Xtrain, ytrain, nBoosts, @decision_tree);% or decision_stump
%     model = adaBoost_RBFSVM(Xtrain, ytrain, nBoosts, @SVM_Kernel);

    % Compute training error
    yhat = model.predict(model,Xtrain);
    error(i, 1) = sum(yhat ~= ytrain) / numel(ytrain);
    
    % Compute test error
    yhat = model.predict(model,Xtest);
    error(i, 2) = sum(yhat ~= ytest) / numel(ytest);
end
save('error/statlog/adaboost_dt_train_test_error_1.mat','error');
plot(error(:, 1), 'b^-');
hold on;
plot(error(:, 2), 'go-');
legend('training error', 'test error', 'b^-', 'go-');
axis([1 nIter 0 1]);
title(['Decision tree, Iteration: ',num2str(nIter), ', # of Base Learner: ', num2str(nBoosts)]);