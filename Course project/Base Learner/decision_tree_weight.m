function t = decision_tree_weight(X, y, z, options)
    % Builds a decision tree to predict y from X.  The tree is grown by
    % recursively splitting each node using the feature which gives the best
    % information gain until the leaf is consistent or all inputs have the same
    % feature values.
    %
    % X is an nxm matrix, where n is the number of points and m is the
    % number of features.
    % y is an nx1 vector of classes
    % cols is a cell-vector of labels for each feature
    %
    % RETURNS t, a structure with 4 entries:
    % t.p is a vector with the index of each node's parent node
    % t.inds is the rows of X in each node (non-empty only for leaves)
    % t.feature is the index of the feature used to split the node
    % t.loc is the value for the feature used to split the node

    % get training set(2/3) and validation set(1/3)
    N = size(X, 1);
    ind = randperm(N, round(N / 3));
    Xvld = X(ind, :);
    yvld = y(ind);
    index = 1 : N;
    index(ind) = [];
    X = X(index, :);
    y = y(index);
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
    options.dt_type = 'C4.5'; % 'C4.5' or 'ID3'
    z = 1 - z;
    t = split_node(X, y, z, t, 1, options);
	t = prune(t, Xvld, yvld, y);
    
    % labels of the leaf nodes
    node_label = NaN * ones(numel(t.p), 1);
    for i = 1 : numel(t.p)
        if(~isempty(t.inds{i}))
            node_label(i) = mode(y(t.inds{i}));
        end
    end
    t.node_label = node_label;
    t.predict = @predict;
end

function yhat = predict(model, X, y)
    if(nargin == 2)
        node_label = model.node_label;
    elseif(nargin == 3)
        inds = model.inds;
    end
    yhat = zeros(size(X, 1), 1);
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
        if(nargin == 2)
            yhat(i) = node_label(node);
        elseif(nargin == 3)
            yhat(i) = mode(y(inds{node}));
        end
    end
end

%% Recursively splits node based on InfoGain(ID3) or InfoRatio(C4.5)
function t = split_node(X, y, z, t, node, options)
    inds = t.inds;
    p = t.p;
    feature = t.feature;
    loc = t.loc;
    dt_type = options.dt_type;

    % Check if the current leaf is consistent
    if numel(unique(y(inds{node}))) == 1
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
    curr_y = y(inds{node});
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
        H = ent(curr_y, curr_z);
        H_cond = zeros(1, size(bin_mat, 2));
        splitI = zeros(1, size(bin_mat, 2));
        for j = 1:size(bin_mat,2)
            H_cond(j) = cond_ent(curr_y, bin_mat(:,j), curr_z);
            % SplitInfo, C4.5
            if(strcmp(dt_type, 'C4.5'))
                splitI(j) = splitInfo(bin_mat(:,j), curr_z);
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
    t = split_node(X, y, z, t, n+1, options);
    t = split_node(X, y, z, t, n+2, options);
end

%% entropy
function result = ent(y, z)
    % Calculates the entropy of a vector of values
    
    % Calculate the probability
    nLabel = unique(y);
    prob = zeros(numel(nLabel), 1);
    for c = 1 : numel(nLabel)
        prob(c) = sum(z(y == nLabel(c))) / sum(z);
    end
    
    % Get entropy
    result = -sum(prob .* log2(prob));
end

%% conditional entropy
function result = cond_ent(y, X, z)
    % Calculates the conditional entropy of y given x
    result = 0;
    
    tab = tabulate(X);
    % Remove zero-entries
    tab = tab(tab(:,3)~=0,:);

    for i = 1 : size(tab,1)
        % Get entropy for y values where x is the current value
        H = ent(y(X == tab(i,1)), z(X == tab(i,1)));
        % Get probability
        prob = sum(z(X == tab(i,1))) / sum(z);
%         prob = tab(i, 3) / 100;
        % Combine
        result = result + prob * H;
    end
end

%% SplitGain
function splitI = splitInfo(y, z)
    splitI = ent(y, z);
end

%% Prune(really important step, otherwise you get CRAZY decision tree)
function t = prune(t, Xvld, yvld, y)
    feature = zeros(numel(t.p), 1);
    loc = zeros(numel(t.p), 1);
    feature(1 : numel(t.feature)) = t.feature;
    loc(1 : numel(t.loc)) = t.loc;
    t.feature = feature;
    t.loc = loc;
    node_level = ones(numel(t.p), 1);
    for i = 1 : numel(t.p)
        node = i;
        while(t.p(node) ~= 0)
            node_level(i) = node_level(i) + 1;
            node = t.p(node);
        end
    end
    
    node = ones(numel(t.p), 1);
    non_leaf = sort(unique(t.p(t.p ~= 0)), 'descend');
    node(non_leaf) = 0;% 0: non-leaf, 1: leaf
    node_check = zeros(numel(t.p), 1);
    
    while(sum(node_check(node == 1)) < numel(node_check(node == 1)))
        max_level = max(node_level(node == 1 & node_check == 0));% unchecked leaf nodes
        if(isempty(max_level))
            break;
        end
        candi_leaf = find(node_level == max_level & node_check == 0);% unchecked leaf nodes with maximum level, may be more than 2
        inds = t.inds;p = t.p;feature = t.feature;loc = t.loc;
        parent = p(candi_leaf(1));
        curr_leaf = (p == parent);
        leaf_ind = getLeafInd(t, parent);
        child_ind = getChildInd(t, parent);
        inds{parent} = sort(leaf_ind);
        inds(child_ind) = [];
        p(child_ind) = -1;
        p = reconstruct_tree(p);
        feature(parent) = 0;
        feature(child_ind) = [];
        loc(parent) = 0;
        loc(child_ind) = [];
        tmp.inds = inds;tmp.p = p;tmp.feature = feature;tmp.loc = loc;

        yorg = predict(t, Xvld, y);
        ytmp = predict(tmp, Xvld, y);
        if(sum(yorg ~= yvld) >= sum(ytmp ~=yvld))
            t = tmp;
            node(parent) = 1;
            node(child_ind) = [];
            node_level(child_ind) = [];
            node_check(child_ind) = [];
        else
            node_check(curr_leaf) = 1;
        end
    end
end

function leaf_ind = getLeafInd(t, parent)
    leaf_ind = [];
    node = parent;
    if(isempty(t.inds{node}))
        node = find(t.p == node);
        leaf_ind = [leaf_ind, getLeafInd(t, node(1))];
        leaf_ind = [leaf_ind, getLeafInd(t, node(2))];
    else
        leaf_ind = t.inds{node};
    end
end

function child_ind = getChildInd(t, parent)
    child_ind = parent + 1 : numel(t.p);
    for i = child_ind
        node = i;
        while ~(node == parent || node == 0)
            node = t.p(node);
        end
        if(node == 0)
            child_ind(child_ind == i) = -1;
        end
    end
    child_ind(child_ind < 0) = [];
end

function ptmp = reconstruct_tree(p)
    ptmp = zeros(numel(p(p >= 0)), 1);
    j = 1;
    for i = 1 : numel(p)
        if(p(i) >= 0)
            ptmp(j) = p(i) - sum(p((1 : p(i) - 1)) < 0);
            j = j + 1;
        end
    end
end