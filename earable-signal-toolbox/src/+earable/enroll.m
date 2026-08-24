function model = enroll(features, subject)
if size(features, 1) ~= numel(subject)
    error('earable:enroll:Length', 'Labels and feature rows must match.');
end

subjects = unique(subject(:))';
centers = zeros(numel(subjects), size(features, 2));
for i = 1:numel(subjects)
    centers(i, :) = median(features(subject == subjects(i), :), 1);
end

spread = std(features, 0, 1);
spread(spread < 1e-5) = 1;
model = struct('subjects', subjects, 'centers', centers, 'spread', spread);
end
