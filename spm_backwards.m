
function [L] = spm_backwards(O,Q,A,B,u,t,T)
% Backwards smoothing to evaluate posterior over initial states
%--------------------------------------------------------------------------
L     = Q{t,2};
p     = 1; 
for timestep = (t + 1):T
    
    % belief propagation over hidden states
    %------------------------------------------------------------------
    

    p    = B{2}(:,:,1)*p;
    
    for state = 1:numel(L)
        % and accumulate likelihood
        %------------------------------------------------------------------
        for g = 3:3
           % possible_states = O{g,timestep}*A{g}(:,:);
           obs = find(cumsum(O{g,timestep})>= rand,1);
           temp = A{g}(obs,:,:);
           temp = permute(temp,[3,2,1]);
           temp = temp*Q{timestep,1}';
           aaa = temp'*p(:,state);
           L(state) = L(state).*aaa;
          
        end
    end
end

% marginal distribution over states
%--------------------------------------------------------------------------
L     = spm_norm(L(:));
end