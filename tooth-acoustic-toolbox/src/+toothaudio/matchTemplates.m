function scores = matchTemplates(model, x, gesture)
scores = zeros(size(x, 1), numel(model.users));

for n = 1:size(x, 1)
    gj = find(model.gestures == gesture(n), 1);
    if isempty(gj)
        error('toothaudio:matchTemplates:Gesture', 'Gesture %g was not enrolled.', gesture(n));
    end
    for u = 1:numel(model.users)
        center = reshape(model.centers(u, gj, :), 1, []);
        delta = (x(n, :) - center) ./ model.spread;
        % cap one wild feature from taking over a whole score
        delta = min(abs(delta), 8);
        scores(n, u) = exp(-sqrt(mean(delta.^2)));
    end
end
end
