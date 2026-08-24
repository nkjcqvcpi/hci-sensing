function y = fixedLength(x, target)
x = x(:);
target = round(target);
if numel(x) >= target
    first = floor((numel(x)-target)/2) + 1;
    y = x(first:first+target-1);
    return
end

before = floor((target-numel(x))/2);
after = target-numel(x)-before;
y = [zeros(before, 1); x; zeros(after, 1)];
end
