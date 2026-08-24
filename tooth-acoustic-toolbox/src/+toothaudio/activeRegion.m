function [event, bounds] = activeRegion(x, cfg)
x = x(:);
energySpan = max(4, round(0.012 * cfg.sampleRate));
e = conv(x.^2, ones(energySpan, 1)/energySpan, 'same');

[~, peak] = max(e);
half = floor(round(cfg.eventSeconds * cfg.sampleRate) / 2);
first = peak - half;
last = first + 2*half;
bounds = [max(first, 1), min(last, numel(x))];
event = toothaudio.internal.fixedLength(x(bounds(1):bounds(2)), 2*half+1);
end
