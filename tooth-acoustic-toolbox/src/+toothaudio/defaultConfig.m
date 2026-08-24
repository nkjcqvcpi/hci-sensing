function cfg = defaultConfig
cfg.sampleRate = 16000;
cfg.clipSeconds = 0.90;
cfg.eventSeconds = 0.38;
cfg.frameSeconds = 0.024;
cfg.hopSeconds = 0.010;
cfg.bandEdges = [120 260 430 680 1050 1550 2250 3150 4300 5800 7600];

cfg.userCount = 7;
cfg.gestureCount = 4;
cfg.takeCount = 6;
cfg.enrollmentTakes = 3;
cfg.randomSeed = 211;
cfg.thresholds = linspace(0.02, 0.98, 241);
end
