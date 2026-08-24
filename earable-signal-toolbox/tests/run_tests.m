function run_tests
cfg = earable.defaultConfig();
cfg.durationSeconds = 5;
cfg.subjectCount = 3;
cfg.sessionCount = 3;
cfg.enrollmentSessions = 2;

testPreprocessing(cfg);
testEndToEnd(cfg);
fprintf('earable tests passed\n');
end

function testPreprocessing(cfg)
a = earable.synthesizeSession(cfg, 2, 1);
b = earable.synthesizeSession(cfg, 2, 1);
assert(isequaln(a.samples, b.samples));

clean = earable.preprocess(a, cfg);
assert(all(isfinite(clean.samples), 'all'));
[frames, starts] = earable.frameSignal(clean, cfg);
assert(size(frames, 3) == numel(starts));
assert(size(frames, 2) == 8);
end

function testEndToEnd(cfg)
data = earable.buildDataset(cfg);
train = data.session <= cfg.enrollmentSessions;
model = earable.enroll(data.features(train, :), data.subject(train));
s = earable.score(model, data.features(~train, :));
r = earable.evaluate(s, data.subject(~train), model.subjects, cfg.thresholds);

assert(all(size(s) == [nnz(~train), cfg.subjectCount]));
assert(isfinite(r.eer) && r.eer >= 0 && r.eer <= 1);
assert(all(r.far >= 0 & r.far <= 1));
end
