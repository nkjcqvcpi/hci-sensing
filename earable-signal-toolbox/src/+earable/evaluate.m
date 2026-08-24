function report = evaluate(scores, truth, enrolledSubjects, thresholds)
truth = truth(:);
if size(scores, 1) ~= numel(truth)
    error('earable:evaluate:Length', 'Truth and score rows do not match.');
end

genuine = zeros(size(truth));
impostor = zeros(size(truth));
for i = 1:numel(truth)
    own = find(enrolledSubjects == truth(i), 1);
    if isempty(own)
        error('earable:evaluate:UnknownSubject', 'Subject %g was not enrolled.', truth(i));
    end
    genuine(i) = scores(i, own);
    other = scores(i, :);
    other(own) = [];
    impostor(i) = max(other);
end

far = arrayfun(@(t) mean(impostor >= t), thresholds);
frr = arrayfun(@(t) mean(genuine < t), thresholds);
[~, idx] = min(abs(far - frr));

report.thresholds = thresholds;
report.far = far;
report.frr = frr;
report.balancedAccuracy = 1 - 0.5 * (far + frr);
report.eer = 0.5 * (far(idx) + frr(idx));
report.eerThreshold = thresholds(idx);
report.genuineScores = genuine;
report.impostorScores = impostor;
end
