function session = synthesizeSession(cfg, subjectId, sessionId)
% repeatable but session noise is kept separate
rng(cfg.randomSeed + 911 * subjectId + 43 * sessionId, 'twister');

n = round(cfg.durationSeconds * cfg.sampleRate);
dt = (1 / cfg.sampleRate) .* (1 + 0.018 * randn(n, 1));
time = cumsum(max(dt, 0.25 / cfg.sampleRate));
time = time - time(1);

f0 = 0.55 + 0.035 * subjectId;
f1 = 1.25 + 0.021 * mod(subjectId, 4);
phase = 0.4 * subjectId + 0.12 * sessionId;

carrier = [sin(2*pi*f0*time + phase), ...
    cos(2*pi*f0*time - 0.6*phase), ...
    sin(2*pi*f1*time + 0.2*phase)];
detail = [sin(2*pi*(2.1*f0)*time), cos(2*pi*(1.7*f1)*time), ...
    sin(2*pi*(f0+f1)*time)];

mix = eye(3) + 0.025 * subjectId * [0 1 -1; -1 0 1; 1 -1 0];
acc = carrier * mix + 0.22 * detail + 0.08 * randn(n, 3);
gyro = 0.7 * detail * mix' + 0.16 * carrier + 0.07 * randn(n, 3);

% slow fit/orientation changes
drift = 0.10 * sin(2*pi*0.045*time + rand(1, 6));
samples = [acc gyro] + drift;

dropCount = max(1, round(0.004 * n));
samples(randperm(n, dropCount), :) = NaN;

session = struct('time', time, 'samples', samples, ...
    'subject', subjectId, 'session', sessionId);
end
