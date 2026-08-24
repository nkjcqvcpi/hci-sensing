function report = assess(scores, truth, users, thresholds)
genuine = zeros(numel(truth), 1);
impostor = zeros(numel(truth), 1);

for i = 1:numel(truth)
    own = find(users == truth(i), 1);
    if isempty(own)
        error('toothaudio:assess:User', 'Query user was not enrolled.');
    end
    genuine(i) = scores(i, own);
    r = scores(i, :);
    r(own) = [];
    impostor(i) = max(r);
end

far = zeros(size(thresholds));
frr = zeros(size(thresholds));
for i = 1:numel(thresholds)
    far(i) = mean(impostor >= thresholds(i));
    frr(i) = mean(genuine < thresholds(i));
end
[~, k] = min(abs(far-frr));

report.thresholds = thresholds;
report.far = far;
report.frr = frr;
report.eer = mean([far(k), frr(k)]);
report.eerThreshold = thresholds(k);
report.bestBalancedAccuracy = max(1 - (far+frr)/2);
report.genuineScores = genuine;
report.impostorScores = impostor;
end
