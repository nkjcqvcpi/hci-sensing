function clean = preprocess(session, cfg)
arguments
    session struct
    cfg struct
end

t = session.time(:);
x = double(session.samples);
if size(x, 1) ~= numel(t) || size(x, 2) < 3
    error('earable:preprocess:Shape', 'Expected N timestamps and an N-by-C sample matrix.');
end
if any(diff(t) <= 0)
    [t, order] = sort(t);
    x = x(order, :);
    [t, keep] = unique(t, 'stable');
    x = x(keep, :);
end

for k = 1:size(x, 2)
    good = isfinite(x(:, k));
    if nnz(good) < 2
        error('earable:preprocess:MissingChannel', 'Channel %d has too few samples.', k);
    end
    x(~good, k) = interp1(t(good), x(good, k), t(~good), 'linear', 'extrap');
end

uniformTime = (t(1):1/cfg.sampleRate:t(end))';
x = interp1(t, x, uniformTime, 'linear');
span = max(3, round(cfg.driftWindowSeconds * cfg.sampleRate));
x = x - earable.internal.movingAverage(x, span);
[x, center, scale] = earable.internal.robustNormalize(x, cfg.clipLevel);

if size(x, 2) >= 6
    x = [x sqrt(sum(x(:, 1:3).^2, 2)) sqrt(sum(x(:, 4:6).^2, 2))];
end

clean = struct('time', uniformTime, 'samples', x, ...
    'center', center, 'scale', scale);
end
