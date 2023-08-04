function kl = kldir(a,b)
% Check for matching dimensions of input matrices
if ~isequal(size(a), size(b))
    error('Input matrices must have the same dimensions.');
end

% Compute KL divergence using element-wise multiplication, sum, and logarithms
kl = sum(a .* log(a./b), 'all');

% Check for NaN or Inf values in kl
if ~isfinite(kl)
    kl = realmax('double');
end
end

%function kl = kldir(a,b)
%kl = sum(a.*(log(a)-log(b)),'all');
%end