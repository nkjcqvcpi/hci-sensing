function [frames, startTime] = frameSignal(clean, cfg)
x = clean.samples;
win = round(cfg.windowSeconds * cfg.sampleRate);
hop = round(cfg.hopSeconds * cfg.sampleRate);
if size(x, 1) < win
    error('earable:frameSignal:ShortInput', 'Recording is shorter than one frame.');
end

starts = 1:hop:(size(x, 1) - win + 1);
frames = zeros(win, size(x, 2), numel(starts));
for i = 1:numel(starts)
    idx = starts(i):(starts(i) + win - 1);
    frames(:, :, i) = x(idx, :);
end
startTime = clean.time(starts);
end
