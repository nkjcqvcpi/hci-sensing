function d = describe(event, cfg)
x = event(:);
frameN = round(cfg.frameSeconds * cfg.sampleRate);
hop = round(cfg.hopSeconds * cfg.sampleRate);
starts = 1:hop:(numel(x)-frameN+1);
if isempty(starts)
    error('toothaudio:describe:ShortEvent', 'Event is shorter than one analysis frame.');
end

window = 0.5 - 0.5*cos(2*pi*(0:frameN-1)'/(frameN-1));
bandRows = zeros(numel(starts), numel(cfg.bandEdges)-1);
centroid = zeros(numel(starts), 1);
flatness = zeros(numel(starts), 1);

for i = 1:numel(starts)
    frame = x(starts(i)+(0:frameN-1)) .* window;
    power = abs(fft(frame)).^2;
    power = power(1:floor(frameN/2)+1);
    freq = (0:numel(power)-1)' * cfg.sampleRate / frameN;
    bandRows(i, :) = toothaudio.internal.bandLogEnergy(power, freq, cfg.bandEdges);
    centroid(i) = sum(freq .* power) / (sum(power)+eps) / (cfg.sampleRate/2);
    flatness(i) = exp(mean(log(power+eps))) / (mean(power)+eps);
end

envelope = conv(abs(x), ones(max(2, round(0.008*cfg.sampleRate)), 1), 'same');
envelope = envelope / (max(envelope)+eps);
q = sort(envelope);
q20 = q(max(1, round(0.2*numel(q))));
q80 = q(max(1, round(0.8*numel(q))));

d = [mean(bandRows, 1), std(bandRows, 0, 1), ...
    mean(centroid), std(centroid), mean(flatness), ...
    sqrt(mean(x.^2)), max(abs(x)), q20, q80, mean(diff(sign(x)) ~= 0)];
end
