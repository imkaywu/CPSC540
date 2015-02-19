function model = SVM1(X, y, z, options) 
    if(nargin == 2)
        z = ones(size(y));
    elseif(nargin == 4)
        ind_1 = options.ind_1;
        alpha_dual = options.alpha_dual;
        X = X(ind_1 == 1, :);
        y = y(ind_1 == 1);
        z = z(ind_1 == 1);
    end
    
    % Training set size
    N = size(X, 1);
    X = [ones(N, 1), X];
    
    % Set regularization parameter
    lambda = 1;
    
    % Set the 'trade-off' term
    C = 1;

    % Initialize dual variables
    alpha = zeros(N,1);
    
    % Set threshold
    thre = 1e-8;

    % Some values used by the dual
    yX = diag(y)*X;
    G = yX*yX';

    while 1
        j = ceil(N * rand);
        alpha(j) = (lambda - alpha' * G(:, j) + alpha(j) * G(j, j)) / G(j, j);
        alpha(j) = min(C * z(j), max(alpha(j), 0));% projection to [0, C * z_i]
        
        w = (1/lambda)*(yX'*alpha);% Convert from dual to primal variables
        P = (lambda / 2) * (w' * w) + C * sum(z .* max(1 - y .* (X * w), 0));% Evaluate primal objective
        D = sum(alpha) - (alpha' * G * alpha) / (2 * lambda);% Evaluate dual objective
        
        if((D - P)^2 < thre)
            break;
        end
    end
    if(nargin == 4)
        alpha_dual(ind_1 == 1) = alpha;
        alpha = alpha_dual;
    end
    model.ind_1 = ind_1;
    model.alpha = alpha;
    model.X = X;
    model.y = y;
    model.predict = @predict;
end

function y = predict(model, X, Kernel, sigma)
    alpha = model.alpha;
%     ind_1 = model.ind_1;
%     alpha = alpha(ind_1 == 1);
    Xtrain = model.X;
    ytrain = model.y;
    
    X = [ones(size(X, 1), 1), standardizeCols(X)];
    if(nargin == 2)
        y = sign(sum(repmat(alpha .* ytrain, [1 size(X, 1)]) .* (Xtrain * X')))';
    elseif(nargin == 4)
        y = sign(sum(repmat(alpha .* ytrain, [1 size(X, 1)]) .* Kernel(Xtrain, X, sigma)) + eps)';
    end
end