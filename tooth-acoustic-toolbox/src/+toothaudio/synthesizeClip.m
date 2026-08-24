function clip = synthesizeClip(cfg, userId, gestureId, takeId)
rng(cfg.randomSeed + 1291*userId + 83*gestureId + 17*takeId, 'twister');
fs = cfg.sampleRate;
n = round(cfg.clipSeconds * fs);
x = 0.006 * randn(n, 1);

eventN = round(cfg.eventSeconds * fs * (0.82 + 0.1*rand));
startAt = round((0.17 + 0.16*rand) * fs);
eventN = min(eventN, n-startAt);
t = (0:eventN-1)' / fs;

base = 360 + 31*userId + 58*gestureId;
spacing = 170 + 9*userId - 7*gestureId;
partials = zeros(eventN, 1);
for k = 1:5
    fk = base + (k-1)*spacing + 5*randn;
    partials = partials + (1/k) * sin(2*pi*fk*t + 2*pi*rand);
end

attack = min(1, t / 0.025);
decay = exp(-(4.2 + 0.35*gestureId) * t / max(t(end), eps));
rough = filter([1 -0.72], 1, randn(eventN, 1));
event = attack .* decay .* (partials + (0.10 + 0.025*gestureId)*rough);

% little impacts for the harder classes
if mod(gestureId, 2) == 0
    p = round(linspace(0.12, 0.72, gestureId+1) * eventN);
    p = p(p > 0 & p <= eventN);
    event(p) = event(p) + 0.7;
end

event = (0.45 + 0.05*randn) * event / (max(abs(event)) + eps);
idx = startAt + (0:eventN-1);
x(idx) = x(idx) + event;

clip = struct('audio', x, 'sampleRate', fs, 'user', userId, ...
    'gesture', gestureId, 'take', takeId);
end
