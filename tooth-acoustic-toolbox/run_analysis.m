function out = run_analysis(cfg)
if nargin < 1
    cfg = toothaudio.defaultConfig();
end

dataset = toothaudio.buildDataset(cfg);
train = dataset.take <= cfg.enrollmentTakes;

model = toothaudio.fitTemplates(dataset.descriptors(train, :), ...
    dataset.user(train), dataset.gesture(train));
eventScores = toothaudio.matchTemplates(model, ...
    dataset.descriptors(~train, :), dataset.gesture(~train));

queryUser = dataset.user(~train);
queryTake = dataset.take(~train);
groupId = queryUser + cfg.userCount * (queryTake - 1);
[scores, truth] = toothaudio.fuseGroups(eventScores, groupId, queryUser, 'logmean');
report = toothaudio.assess(scores, truth, model.users, cfg.thresholds);

fprintf('events: %d, fused queries: %d\n', size(dataset.descriptors, 1), size(scores, 1));
fprintf('equal-error estimate: %.3f\n', report.eer);
out = struct('config', cfg, 'dataset', dataset, 'model', model, ...
    'eventScores', eventScores, 'scores', scores, 'truth', truth, 'report', report);
end
