
function G = G_epistemic_value(A,s)
    
% auxiliary function for Bayesian suprise or mutual information
% FORMAT [G] = spm_MDP_G(A,s)
%
% A   - likelihood array (probability of outcomes given causes)
% s   - probability density of causes

% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_MDP_G.m 7306 2018-05-07 13:42:02Z karl $

% probability distribution over the hidden causes: i.e., Q(s)

qx = spm_cross(s); % this is the outer product of the posterior over states
                   % calculated with respect to itself

% accumulate expectation of entropy: i.e., E[lnP(o|s)]
G     = 0;
qo    = 0;
for i = find(qx > exp(-16))'
    % probability over outcomes for this combination of causes
    po   = 1;
    for g = 1:numel(A)
        po = spm_cross(po,A{g}(:,i));
    end
    po = po(:);
    qo = qo + qx(i)*po;
    G  = G  + qx(i)*po'*nat_log(po);
end

% subtract entropy of expectations: i.e., E[lnQ(o)]
G  = G - qo'*nat_log(qo);
    
end 
