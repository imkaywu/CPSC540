function [model] = matLearn_classification_decisionTree(X,y,options)
% matLearn_classification_decisionTree(X,y,options)
%
% Description:



%   - DESCRIPTION HERE!!!!
%



% Options:
%   - None
%
% Authors:
%




root = fitTree(X, y, 0, options);


model.name = 'Decision Tree';
model.predict = @predict;
model.getTree = @getTree;
model.root = root;
end

function [yhat] = predict(model,Xhat)
    [nTest,~] = size(Xhat);
    
    yhat = zeros(nTest,1);
    
    for i = 1 : nTest
        
        currNode = model.root;
        
        while ~isempty(currNode.left) || ~isempty(currNode.right)
            if Xhat(i,currNode.feature) < currNode.threshold
                currNode = currNode.left;
            else
                currNode = currNode.right;
            end
        end
        yhat(i) = currNode.class;
    end
end


function [gain] = infoGain()
    gain = 123;
end


function [node] = fitTree(data, y, depth, options)
    [nData,nFeatures] = size(data);
    
    classes = unique(y);   %get the number of unique classes
    [nClasses,~] = size(classes);   %get the number of unique classes

    classCounts = histc(y, unique(y(:)));   %count each class's occurrence
    classProbs = classCounts / nData;      %calc class probabilities
    
    
    [maxClassProb,maxClassProbIndex] = max(classProbs);
    mostProbClass = y(maxClassProbIndex);
    
    %split based on classification error:
    minErr = inf;
    for j = 1 : nFeatures
        thresholds = [sort(unique(data(:,j)));max(data(:,j))+eps];

        for t = thresholds'
            err = sum(data(y==mostProbClass,j) < t) + sum(data(y~=mostProbClass,j) >= t);

            if err < minErr
                minErr = err;
                selectedFeature = j;
                minThreshold = t;
            end
        end
    end
    
    newNode.feature = selectedFeature;
    newNode.threshold = minThreshold;

    %get the indices of all the values on either side of the threshold
    leftDataIndices = find( data(:,selectedFeature) < minThreshold );
    rightDataIndices = find( data(:,selectedFeature) >= minThreshold );
   
    [sizeLeft,~] = size(leftDataIndices);
    [sizeRight,~] = size(rightDataIndices);
 
    newNode.depth = depth;
    
    %determine if worth splitting
    if ( sizeLeft > 0 ) && ( sizeRight > 0 )
        newNode.left = fitTree(data(leftDataIndices), y(leftDataIndices), depth + 1);
        newNode.right = fitTree(data(rightDataIndices), y(rightDataIndices), depth + 1);
    else
        %all data is on one side of the threshold
        newNode.left = [];
        newNode.right = [];
        newNode.class = mode(y);   %return the most common class label
    end
    
    node = newNode;
end

function [tree] = getTree(model)
    tree = getSubTree(model.root)';
end

function [tree] = getSubTree(node)
    
    tree(1) = node.depth;
    
    if ~isempty(node.left)
        tree = [tree; getSubTree(node.left)];
    end
    
    if ~isempty(node.right)
        tree = [tree; getSubTree(node.right)];
    end

    
end