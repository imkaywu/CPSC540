% clear all
% close all
clear;clc;

%% Load synthetic {Xtrain,ytrain} and {Xtest,ytest}
load iris.mat

X = dataset(:, 1 : (size(dataset, 2) - 1));
y = dataset(:, size(dataset, 2));

options.nModel = 200;
[N,d] = size(X);
validationTrainSize = 0.7;
train_ind = randperm(N, ceil(validationTrainSize * N));
test_ind = 1 : N;
test_ind(train_ind) = [];
%lastTrainIndex = floor(N*validationTrainSize);

Xtrain = X(train_ind,:);
ytrain = y(train_ind);
Xtest = X(test_ind,:);
ytest = y(test_ind);
%% Decision Tree model




% Train decision tree model
%options.nModel = 1000;
[model_bagging] = matLearn_classification_bagging(Xtrain,ytrain,options);

% Test decision tree model
options.dt_type = 'C4.5';
yhat = model_bagging.predict(model_bagging,Xtrain,ytrain,Xtest,options);

% Measure test error
testError = sum(yhat~=ytest)/length(ytest);
fprintf('Averaged absolute test error with %s is: %.3f\n',model_bagging.name,testError);

% Plot model predictions
pointSize = 75;

plotFeature1 = 1;
plotFeature2 = 2;

feature1Vals = Xtest(:,plotFeature1);
feature2Vals = Xtest(:,plotFeature2);

scatter(feature1Vals, feature2Vals, pointSize, yhat);hold on;
incorrect = find(ytest~=yhat);

scatter(feature1Vals(incorrect), feature2Vals(incorrect), pointSize, ytest(incorrect), 'x');

% tree = model_decision_tree.getTree(model_decision_tree);
% treeplot(tree);
% 
% childHandles = get(gca, 'Children'); % get handles to children
% % grab X and Y coords from the second child (the first one is axes)
% xCoord = get(childHandles(2), 'XData');
% yCoord = get(childHandles(2), 'YData');
% 
% text(xCoord, yCoord + 0.02, strtrim(cellstr(num2str(tree'))'),'VerticalAlignment','bottom','HorizontalAlignment','center')