function model = fitTemplates(x, user, gesture)
if size(x, 1) ~= numel(user) || numel(user) ~= numel(gesture)
    error('toothaudio:fitTemplates:Length', 'Input arrays have different lengths.');
end

users = unique(user(:))';
gestures = unique(gesture(:))';
centers = zeros(numel(users), numel(gestures), size(x, 2));

for i = 1:numel(users)
    for j = 1:numel(gestures)
        pick = user == users(i) & gesture == gestures(j);
        if ~any(pick)
            error('toothaudio:fitTemplates:MissingCell', 'Missing user/gesture enrollment cell.');
        end
        centers(i, j, :) = median(x(pick, :), 1);
    end
end

spread = std(x, 0, 1);
spread(spread < 1e-6) = 1;
model = struct('users', users, 'gestures', gestures, ...
    'centers', centers, 'spread', spread);
end
