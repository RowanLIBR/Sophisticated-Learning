function [G,P, short_term_memory, best_actions] = tree_search_frwd_SI(short_term_memory, O, P, a, A, y, B,b, t, T, N, t_food,t_water,t_sleep, true_t, chosen_action, true_t_food, true_t_water, true_t_sleep, best_actions, learning_weight, novelty_weight, epistemic_weight, preference_weight)
  
    G = 0.02;
    P_prior = P;
    P = calculate_posterior(P,y,O,t);
    bb{2} = normalise_matrix(b{2});
    t_food_approx = round(t_food+1);
    t_water_approx = round(t_water+1);
    t_sleep_approx = round(t_sleep+1);
    num_factors = 2;
    if t > true_t 
     novelty = 0;
    
   
     for timey = t:t
       if timey ~= t
            L = spm_backwards(O,P,A,bb,chosen_action,timey,t);
        else
            L = P{t,2};                
        end
        
        LL{2} = L;
        LL{1} = P{timey,1};
        a_prior  = a{2};
        for modality = 2:2
          a_learning = O(modality,timey)';
          for  factor = 1:num_factors
             a_learning = spm_cross(a_learning, LL{factor});
          end
          a_learning = a_learning.*(a{modality} > 0);
          a_learning_weighted = a_learning;
          a_learning_weighted(2:end,:) = learning_weight*a_learning(2:end,:);
          a_learning_weighted(1,:) = a_learning(1,:);
          a_temp = a_prior + a_learning_weighted;  
        end
        
        w = kldir(normalise(a_temp(:)),normalise(a_prior(:)));
        novelty = novelty + w;          
    end
       
    

    % Add epistemic term (see EFE equation)
    epi = G_epistemic_value(y,P_prior(t,:)');
   
    % Add novelty to term (see EFE equation)
     G = G + novelty_weight*novelty;
    G = G + epistemic_weight*epi;
    for modality = 2:2
        if modality == 2
            C = determineObservationPreference(t_food, t_water, t_sleep);
            %reduce preference precision
            C{modality} = C{modality}/preference_weight;
        end
        if modality == 2
            % add extrinsic term (see EFE equation)
            extrinsic = O{2,t}*C{2}';
            G = G + extrinsic;
        end
    end

         
         t_food = round(t_food*(1-O{2,t}(2)))+1;
         t_water = round(t_water*(1-O{2,t}(3)))+1;
           t_sleep = round(t_sleep*(1-O{2,t}(4)))+1;
         t_food_approx = t_food;
         t_water_approx = t_water;
         t_sleep_approx = t_sleep;

    end      
        
    if t < N
        
       actions = randperm(5);
       efe = [0,0,0,0,0];
       for action = actions
            
            Q{1,action} = (B{1}(:,:,action)*P{t,1}')';
            Q{2,action} = (bb{2}(:,:,1)*P{t,2}');
            s = Q(:,action);
            qs = spm_cross(s);
            qs = qs(:);
            % only consider relatively likely states
            likely_states = find(qs > 1/8);
            if isempty(likely_states)
                threshold = 1/numel(qs)*1/numel(qs);
                likely_states = find(qs > (1/numel(qs)-threshold));
            end
            % for each of those likely states
            for state = likely_states(:)'
                 if short_term_memory(t_food_approx,t_water_approx,t_sleep_approx,state) ~= 0 
                     sh = short_term_memory(t_food_approx,t_water_approx,t_sleep_approx, state);
                     K(state) = sh;
                 else
                    for modal = 1:numel(A) 
                      O{modal,t+1} = normalise(y{modal}(:,state)');
                    end   
                    % prior over next states given transition function
                    % (calculated earlier)
                    P{t+1,1} = Q{1, action};
                    P{t+1,2} = Q{2, action};
                    chosen_action(t) = action;
                    
                    % recursively move to the next node (likely state) of
                    % the tree
                    [expected_free_energy, P, short_term_memory, best_actions] = tree_search_frwd_SI(short_term_memory, O, P, a, A, y, B,b, t+1, T, N, t_food_approx,t_water_approx,t_sleep_approx, true_t, chosen_action, true_t_food, true_t_water, true_t_sleep, best_actions, learning_weight, novelty_weight, epistemic_weight, preference_weight);
                    S = max(expected_free_energy);
                    K(state) = S;
                    short_term_memory(t_food_approx,t_water_approx,t_sleep_approx,state) = S;
                 end
                 
               
             end
                
                
        
        action_fe = K(likely_states)*qs(likely_states);
        efe(action) = efe(action) + 0.7*action_fe;
            
      end
        [maxi, chosen_action] = max(efe);
        G = G + maxi;
        best_actions = [chosen_action best_actions];
   end