function y = movingAverage(x, span)
span = max(1, round(span));
kernel = ones(span, 1) / span;
y = zeros(size(x));
for c = 1:size(x, 2)
    padded = [repmat(x(1, c), floor(span/2), 1); x(:, c); ...
        repmat(x(end, c), ceil(span/2)-1, 1)];
    y(:, c) = conv(padded, kernel, 'valid');
end
end
