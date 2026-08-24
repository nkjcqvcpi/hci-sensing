function [fused, truth, groups] = fuseGroups(scores, groupId, labels, method)
if nargin < 4
    method = 'mean';
end
groups = unique(groupId(:), 'stable');
fused = zeros(numel(groups), size(scores, 2));
truth = zeros(numel(groups), 1);

for i = 1:numel(groups)
    pick = groupId == groups(i);
    labelSet = unique(labels(pick));
    if numel(labelSet) ~= 1
        error('toothaudio:fuseGroups:MixedLabels', 'A score group contains mixed labels.');
    end
    truth(i) = labelSet;
    block = scores(pick, :);
    switch lower(method)
        case 'mean'
            fused(i, :) = mean(block, 1);
        case 'logmean'
            fused(i, :) = exp(mean(log(max(block, eps)), 1));
        case 'median'
            fused(i, :) = median(block, 1);
        otherwise
            error('toothaudio:fuseGroups:Method', 'Unknown fusion method: %s', method);
    end
end
end
