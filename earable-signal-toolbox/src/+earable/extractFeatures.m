function features = extractFeatures(frames, sampleRate)
[n, channels, count] = size(frames);
features = zeros(count, channels * 7);
freq = (0:floor(n/2))' * sampleRate / n;

for i = 1:count
    row = zeros(channels, 7);
    for c = 1:channels
        v = frames(:, c, i);
        spec = abs(fft(v));
        spec = spec(1:numel(freq)).^2;
        totalPower = sum(spec) + eps;
        centroid = sum(freq .* spec) / totalPower;
        low = sum(spec(freq <= 3.0));
        high = sum(spec(freq > 3.0 & freq <= 12.0));

        row(c, :) = [mean(v), std(v, 1), sqrt(mean(v.^2)), ...
            max(v)-min(v), mean(abs(diff(sign(v))) > 0), ...
            centroid / (sampleRate/2), log1p(low) - log1p(high)];
    end
    features(i, :) = reshape(row', 1, []);
end
end
