function y = condition(x)
x = double(x(:));
if isempty(x) || any(~isfinite(x))
    error('toothaudio:condition:Input', 'Audio must be a finite nonempty vector.');
end

x = x - median(x);
y = filter([1 -0.94], 1, x);
limit = 6 * median(abs(y - median(y))) + eps;
y = min(max(y, -limit), limit);
r = sqrt(mean(y.^2));
if r > eps
    y = y / r;
end
end
