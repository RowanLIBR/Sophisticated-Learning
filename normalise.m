function x = normalise(array)
x = array/(sum(array));
if isnan(x)
    x = ones(numel(x),1);
    x(:) = 1/numel(x);
end
end