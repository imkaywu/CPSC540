function t = decision_tree_weight(X, Y, z, options)
    % Builds a decision tree to predict Y from X.  The tree is grown by
    % recursively splitting each node using the feature which gives the best
    % information gain until the leaf is consistent or all inputs have the same
    % feature values.
    %
    % X is an nxm matrix, where n is the number of points and m is the
    % number of features.
    % Y is an nx1 vector of classes
    % cols is a cell-vector of labels for each feature
    %
    % RETURNS t, a structure with 4 entries:
    % t.p is a vector with the index of each node's parent node
    % t.inds is the rows of X in each node (non-empty only for leaves)
    % t.feature is the index of the feature used to split the node
    % t.loc is the value for the feature used to split the node

    % Create an empty decision tree, which has one node and everything in it
    inds = {1:size(X,1)}; % A cell per node containing indices of all data in that node
    p = 0; % Vector contiaining the index of the parent node for each node
    feature = []; % Feature used to split the node
    loc = []; % value of the feature used to split the node

    t.inds = inds;
    t.p = p;
    t.feature = feature;
    t.loc = loc;

    % Create tree by splitting on the root
    options.dt_type = 'ID3'; % or 'ID3'
    t = split_node(X, Y, z, t, 1, options);
	
    % labels of the leaf nodes
    node_label = NaN * ones(numel(t.p), 1);
    for i = 1 : numel(t.p)
        if(~isempty(t.inds{i}))
            node_label(i) = mode(Y(t.inds{i}));
        end
    end
    t.node_label = node_label;
    t.predict = @predict;
end

function y = predict(model, X)
    node_label = model.node_label;
    y = zeros(size(X, 1), 1);
    for i = 1 : size(X, 1)
        node = 1;
        while(isempty(model.inds{node}))
            node_child = find(model.p == node);
            if(X(i, model.feature(node)) < model.loc(node))
                node = node_child(1);
            else
                node = node_child(2);
            end
        end
        y(i) = node_label(node);
    end
end

%% Recursively splits node based on InfoGain(ID3) or InfoRatio(C4.5)
function t = split_node(X, Y, z, t, node, options)
    inds = t.inds;
    p = t.p;
    feature = t.feature;
    loc = t.loc;
    dt_type = options.dt_type;

    % Check if the current leaf is consistent
    if numel(unique(Y(inds{node}))) == 1
        return;
    end

    % Check if all inputs have the same features
    % We do this by seeing if there are multiple unique rows of X
    if size(unique(X(inds{node},:),'rows'),1) == 1
        return;
    end

    % Otherwise, we need to split the current node on some feature

    best_ig = -inf; %best information gain
    best_feature = 0; %best feature to split on
    best_val = 0; % best value to split the best feature on

    curr_X = X(inds{node},:);
    curr_Y = Y(inds{node});
    curr_z = z(inds{node});
    % Loop over each feature
    for i = 1:size(X,2)
        feat = curr_X(:,i);

        % Deterimine the values to split on
        vals = unique(feat);
        splits = 0.5*(vals(1:end-1) + vals(2:end));
        if numel(vals) < 2
            continue
        end

        % Get binary values for each split value
        bin_mat = double(repmat(feat, [1 numel(splits)]) < repmat(splits', [numel(feat) 1]));

        % Compute the information gains
%         H = ent(curr_Y, curr_z);
        H = ent(curr_Y);
        H_cond = zeros(1, size(bin_mat, 2));
        splitI = zeros(1, size(bin_mat, 2));
        for j = 1:size(bin_mat,2)
            H_cond(j) = cond_ent(curr_Y, bin_mat(:,j), curr_z);
            % SplitInfo, C4.5
            if(strcmp(dt_type, 'C4.5'))
%                 splitI(j) = splitInfo(bin_mat(:,j), curr_z);
                splitI(j) = splitInfo(bin_mat(:,j));
            end
        end

        if(strcmp(dt_type, 'ID3'))
            IG = H - H_cond;
            % Find the best split
            [val, ind] = max(IG);
        elseif(strcmp(dt_type, 'C4.5'))
            gainRatio = (H - H_cond) ./ splitI;
            % Find the best split
            [val, ind] = max(gainRatio);
        end

        if val > best_ig
            best_ig = val;
            best_feature = i;
            best_val = splits(ind);
        end
    end

    % Split the current node into two nodes
    feat = curr_X(:,best_feature);
    feat = feat < best_val;
    inds = [inds; inds{node}(feat); inds{node}(~feat)];
    inds{node} = [];
    p = [p; node; node];
    feature(node) = best_feature;
    loc(node) = best_val;

    t.inds = inds;
    t.p = p;
    t.feature = feature;
    t.loc = loc;

    % Recurse on newly-create nodes
    n = numel(p)-2;
    t = split_node(X, Y, z, t, n+1, options);
    t = split_node(X, Y, z, t, n+2, options);
end

%% entropy
% function result = ent(Y, z)
%     % Calculates the entropy of a vector of values
%     
%     % Calculate the probability
%     nLabel = unique(Y);
%     prob = zeros(numel(nLabel), 1);
%     for c = 1 : numel(nLabel)
%         prob(c) = sum(z(Y == nLabel(c))) / sum(z);
%     end
%     
%     % Get entropy
%     result = -sum(prob .* log2(prob));
% end

%% conditional entropy
% function result = cond_ent(Y, X, z)
%     % Calculates the conditional entropy of y given x
%     result = 0;
%     
%     tab = tabulate(X);
%     % Remove zero-entries
%     tab = tab(tab(:,3)~=0,:);
% 
%     for i = 1 : size(tab,1)
%         % Get entropy for y values where x is the current value
%         H = ent(Y(X == tab(i,1)), z(X == tab(i,1)));
%         % Get probability
%         prob = sum(z(X == tab(i,1))) / sum(z);
% %         prob = tab(i, 3) / 100;
%         % Combine
%         result = result + prob * H;
%     end
% end

%% SplitGain
% function splitI = splitInfo(Y, z)
%     splitI = ent(Y, z);
% end

function result = ent(Y)
    % Calculates the entropy of a vector of values 

    % Get frequency table 
    tab = tabulate(Y);
    prob = tab(:,3) / 100;
    % Filter out zero-entries
    prob = prob(prob~=0);
    % Get entropy
    result = -sum(prob .* log2(prob));
end

function result = cond_ent(Y, X, z)
    % Calculates the conditional entropy of y given x
    result = 0;
    
    tab = tabulate(X);
    % Remove zero-entries
    tab = tab(tab(:,3)~=0,:);

    for i = 1 : size(tab,1)
        % Get entropy for y values where x is the current value
        H = ent(Y(X == tab(i,1)));
        % Get probability
        prob = sum(z(X == tab(i,1))) / sum(z);
%         prob = tab(i, 3) / 100;
        % Combine
        result = result + prob * H;
    end
end

function splitI = splitInfo(Y)
    splitI = ent(Y);
end