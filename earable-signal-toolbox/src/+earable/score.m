function scores = score(model, features)
n = size(features, 1);
k = size(model.centers, 1);
scores = zeros(n, k);

for i = 1:k
    delta = (features - model.centers(i, :)) ./ model.spread;
    d = sqrt(mean(delta.^2, 2));
    scores(:, i) = exp(-d);
end
end
