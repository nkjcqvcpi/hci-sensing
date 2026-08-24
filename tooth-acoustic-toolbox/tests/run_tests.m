function run_tests
cfg = toothaudio.defaultConfig();
cfg.userCount = 3;
cfg.gestureCount = 3;
cfg.takeCount = 4;
cfg.enrollmentTakes = 2;

testAudioPath(cfg);
testAnalysis(cfg);
fprintf('tooth audio tests passed\n');
end

function testAudioPath(cfg)
a = toothaudio.synthesizeClip(cfg, 1, 2, 1);
b = toothaudio.synthesizeClip(cfg, 1, 2, 1);
assert(isequal(a.audio, b.audio));

y = toothaudio.condition(a.audio);
[event, bounds] = toothaudio.activeRegion(y, cfg);
d = toothaudio.describe(event, cfg);
assert(all(isfinite(d)));
assert(numel(event) == 2*floor(round(cfg.eventSeconds*cfg.sampleRate)/2)+1);
assert(bounds(1) >= 1 && bounds(2) <= numel(y));
end

function testAnalysis(cfg)
data = toothaudio.buildDataset(cfg);
train = data.take <= cfg.enrollmentTakes;
model = toothaudio.fitTemplates(data.descriptors(train, :), ...
    data.user(train), data.gesture(train));
s = toothaudio.matchTemplates(model, data.descriptors(~train, :), data.gesture(~train));

groupId = data.user(~train) + cfg.userCount*(data.take(~train)-1);
[fused, truth] = toothaudio.fuseGroups(s, groupId, data.user(~train), 'logmean');
r = toothaudio.assess(fused, truth, model.users, cfg.thresholds);
assert(size(fused, 2) == cfg.userCount);
assert(isfinite(r.eer));
assert(r.eer >= 0 && r.eer <= 1);
end
