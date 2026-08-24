function result = run_pipeline(cfg)
if nargin == 0
    cfg = earable.defaultConfig();
end

data = earable.buildDataset(cfg);
isEnrollment = data.session <= cfg.enrollmentSessions;

model = earable.enroll(data.features(isEnrollment, :), ...
    data.subject(isEnrollment));
scores = earable.score(model, data.features(~isEnrollment, :));
report = earable.evaluate(scores, data.subject(~isEnrollment), ...
    model.subjects, cfg.thresholds);

fprintf('windows: %d\n', size(data.features, 1));
fprintf('equal-error estimate: %.3f at %.3f\n', report.eer, report.eerThreshold);

result = struct('config', cfg, 'model', model, 'scores', scores, ...
    'truth', data.subject(~isEnrollment), 'report', report);
end
