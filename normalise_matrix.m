function m = normalise_matrix(m)
  for i = 1:length(m(1,:))
      m(:,i) = m(:,i)/sum(m(:,i));
  end
end

