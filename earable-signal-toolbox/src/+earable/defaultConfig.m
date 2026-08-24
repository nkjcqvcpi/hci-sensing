function cfg = defaultConfig
cfg.sampleRate = 100;
cfg.durationSeconds = 14;
cfg.subjectCount = 8;
cfg.sessionCount = 5;
cfg.enrollmentSessions = 3;

cfg.windowSeconds = 1.8;
cfg.hopSeconds = 0.6;
cfg.driftWindowSeconds = 0.75;
cfg.clipLevel = 7;
cfg.randomSeed = 73;
cfg.thresholds = linspace(0.05, 0.95, 181);
end
