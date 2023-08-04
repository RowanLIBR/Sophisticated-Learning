function [G,P, short_term_memory, best_actions] = tree_search_frwd(short_term_memory, O, P, a, A,y, B,b, t, T, N, t_food,t_water,t_sleep, true_t, chosen_action, novelty, true_t_food, true_t_water, true_t_sleep, best_actions)
  
    G = 0.02;
    P = calculate_posterior(P,y,O,t);
    bb{2} = normalise_matrix(b{2});
    cur_state = find(cumsum(P{t,1})>=rand,1);
    cur_context = find(cumsum(P{t,2})>=rand,1);
     
        if t_food>35   
            t_food = 35;
        end
    
        if t_water>35    
            t_water = 35;
        end
        if t_sleep>35    
            t_sleep = 35;
        end
        for modality = 2:2
            if modality == 2
                C = determineObservationPreference(t_food, t_water, t_sleep);
                %reduce preference precision
                C{modality} = C{modality};
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

    if t < N %&& t_sleep_approx < 14 && t_water_approx < 10 && t_food_approx < 12
        
        actions = randperm(5);
        cur_state = spm_cross(P(t,:));
        cur_state = find(cumsum(cur_state(:))>=rand,1);
        efe_future = [0,0,0,0,0];
        for action = actions
            if short_term_memory(t_food,t_water,t_sleep,cur_state, action) ~= 0 
                   sh = short_term_memory(t_food, t_water, t_sleep, cur_state, action);
                   %S =  sh;
                   efe_future(action) = sh;
            else
                Q{1,action} = (B{1}(:,:,action)*P{t,1}')';
                Q{2,action} = (bb{2}(:,:,1)*P{t,2}');
                s = Q(:,action);
                qs = spm_cross(s);
                qs = qs(:);
                % only consider relatively likely states
                likely_states = find(qs > 1/8);
%                 if isempty(likely_states)
%                     threshold = 1/numel(qs)*1/numel(qs);
%                     likely_states = find(qs > (1/numel(qs)-threshold));
%                 end
                % for each of those likely states
                for state = likely_states(:)'
                    % check to see if we have already calculated a value for
                    % this state
                    
                        % get distribution over possible observations given
                        % state
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
                        [expected_free_energy, D, short_term_memory, best_actions] = tree_search_frwd(short_term_memory, O, P, a, A,y, B,b, t+1, T, N, t_food_approx,t_water_approx,t_sleep_approx, true_t, chosen_action, 0, true_t_food, true_t_water, true_t_sleep, best_actions);
                        
                        S = max(expected_free_energy);
                        K(state) = S;
                end

            action_fe = K(likely_states)*qs(likely_states);
            efe_future(action) = efe_future(action) + 0.7*action_fe;
            short_term_memory(t_food, t_water, t_sleep,cur_state, action) = 0.7*action_fe;
            end
        end
        [maxi, chosen_action] = max(efe_future);
        G = G + maxi;
        best_actions = [chosen_action best_actions];
    end     
end
