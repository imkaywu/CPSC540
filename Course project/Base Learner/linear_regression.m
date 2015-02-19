function model = linear_regression(X, y, z, options)
%     lambda = 1 / numel(y);
    z = 1 - z;
    w = (X' * diag(z) * X) \ X' * diag(z) * y;
%     w = (X' * X + lambda * eye(size(X, 2))) \ X' * y;
    
    model.w = w;
    model.predict = @predict;
end

function y = predict(model, X)
    y = sign(X * model.w);
    y(y == -1) = 0;
end