function data = buildDataset(cfg)
featureBlocks = cell(cfg.subjectCount, cfg.sessionCount);
subjectBlocks = cell(size(featureBlocks));
sessionBlocks = cell(size(featureBlocks));

for s = 1:cfg.subjectCount
    for r = 1:cfg.sessionCount
        raw = earable.synthesizeSession(cfg, s, r);
        clean = earable.preprocess(raw, cfg);
        frames = earable.frameSignal(clean, cfg);
        f = earable.extractFeatures(frames, cfg.sampleRate);
        featureBlocks{s, r} = f;
        subjectBlocks{s, r} = repmat(s, size(f, 1), 1);
        sessionBlocks{s, r} = repmat(r, size(f, 1), 1);
    end
end

data.features = vertcat(featureBlocks{:});
data.subject = vertcat(subjectBlocks{:});
data.session = vertcat(sessionBlocks{:});
end
