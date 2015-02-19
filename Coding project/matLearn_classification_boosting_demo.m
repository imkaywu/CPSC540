clear; clc
%% dataset 1: binary
% load('binary.mat');
% [X,~,~] = standardizeCols(X);
% N = size(X,1);
%% dataset 2: iris
load('iris.mat');
X = dataset(:, 1 : (size(dataset, 2) - 1));
y = dataset(:, size(dataset, 2));

% Training
options.nBoosts = 20;
options.classifier = 'decision tree';%or 'decision stump'
if(strcmp(options.classifier, 'decision tree'))
    options.dt_type = 'C4.5'; % or 'ID3'
end
model = matLearn_classification_boosting(X, y, options);

% Testing
yhat = model.predict(model, X);
trainError = sum(y ~= yhat) / size(y, 1);

fprintf('training error with %s is: %.3f\n',options.classifier, trainError);