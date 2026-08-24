function data = buildDataset(cfg)
total = cfg.userCount * cfg.gestureCount * cfg.takeCount;
rows = cell(total, 1);
user = zeros(total, 1);
gesture = zeros(total, 1);
take = zeros(total, 1);
bounds = zeros(total, 2);

at = 0;
for u = 1:cfg.userCount
    for g = 1:cfg.gestureCount
        for r = 1:cfg.takeCount
            at = at + 1;
            clip = toothaudio.synthesizeClip(cfg, u, g, r);
            y = toothaudio.condition(clip.audio);
            [event, eventBounds] = toothaudio.activeRegion(y, cfg);
            bounds(at, :) = eventBounds;
            rows{at} = toothaudio.describe(event, cfg);
            user(at) = u;
            gesture(at) = g;
            take(at) = r;
        end
    end
end

data = struct('descriptors', vertcat(rows{:}), 'user', user, ...
    'gesture', gesture, 'take', take, 'bounds', bounds);
end
