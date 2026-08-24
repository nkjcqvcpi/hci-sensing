function [z, center, scale] = robustNormalize(x, clipLevel)
center = median(x, 1);
deviation = abs(x - center);
scale = 1.4826 * median(deviation, 1);
scale(scale < 1e-8) = 1;
z = (x - center) ./ scale;
z = min(max(z, -clipLevel), clipLevel);
end
