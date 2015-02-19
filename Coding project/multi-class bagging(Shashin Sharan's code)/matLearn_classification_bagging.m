function [model] = matLearn_classification_bagging(X,y,options)
% matLearn_classification_bagging(X,y,options)
%
% Description:
%    - Classification based on the average prediction among models fit to
% bootstrap samples
% 
% Options
%    - Specify the number of bootstrap samples
%    - Specify the input model that need to be bagged
%
% Authors:
%  - Shashin Sharan (2014)

[nTrain,~] = size(X);
model.nTrain = nTrain;
model.nbs = options.nModel; %no. of bootstrap samples
model.name = 'Bagging';
%model.subroutine = @Decisiontree; % The model which is to be bagged.Input models can change from problem to problem.
model.predict = @predict;

end

function [yhat] = predict(model,X,y,Xhat,options)
% Prediction of class using the bagged classifier

nbs = model.nbs; % Total number of bootstrap samples
C = sort(unique(y)); % Unique classes are sorted
ybshat = bag(model,X,y,Xhat,options); % Classes predicted for each of the bootstrap sample


[nTest,~] = size(Xhat);

for m = 1:nTest %no. of test data i.e. number of rows in the test data
    
    counts = zeros(size(C,1),1); % Create a null vector to store the number of votes for each class..
    % corresponding to each bootstrap sample
    
    for k = 1:size(C,1)
       
        
        for j = 1:nbs
        
        counts(k) = counts(k) + sum(ybshat(m,j) == C(k)); % For every class k, number of votes is counted
        
        end
        
    end
    
    [max_vote, max_index] = max(counts); % Total no. of votes and index corresponding to the label with ...
    %maximum number of votes is extracted
    
    
    yhat(m) = C(max_index); % The test data corresponding to m'th row is assigned class label...
    % which correspond to the maximum voted label
    
end

yhat = yhat';

end

function [ybshat] = bag(model,X,y,Xhat,options)

[nTrain,~] = size(X);

nbs = model.nbs;

for j = 1:nbs
    for i = 1:ceil(nTrain*0.63)
        ind = ceil(rand*nTrain);
        Xbs(i,:) = X(ind, :);
        ybs(i,1) = y(ind);
    end
    
    % Training of each bootstrap sample
    
    options = [];
    
%     submodel = matLearn_classification_decisionTree(Xbs,ybs,options);
    options.dt_type = 'C4.5';
    submodel = decision_tree(Xbs, ybs, options);
    
%     ybshat(:,j) = submodel.predict(submodel,Xhat); % corresponding to each predictor based on each bootstrap...
    ybshat(:, j) = submodel.predict(submodel, Xhat);

    % a class label is predicted and stored in columns of yhat
    % corresponding to the bootstrap sample
    
end

end
