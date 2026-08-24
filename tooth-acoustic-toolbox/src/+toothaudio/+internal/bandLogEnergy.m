function row = bandLogEnergy(power, freq, edges)
row = zeros(1, numel(edges)-1);
for b = 1:numel(row)
    if b == numel(row)
        pick = freq >= edges(b) & freq <= edges(b+1);
    else
        pick = freq >= edges(b) & freq < edges(b+1);
    end
    row(b) = log(sum(power(pick)) + eps);
end
end
