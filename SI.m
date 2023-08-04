%function [] = SI(seed)
%rng(str2double(seed));
clear
previous_positions = [];
previous_positions(end+1) = 55; % hill


hill_1 = 55;
true_food_source_1 = 71;
true_food_source_2 = 43;
true_food_source_3 =57;
true_food_source_4 = 78;
true_water_source_1 = 73;
true_water_source_2 = 33;
true_water_source_3 = 48;
true_water_source_4 = 67;
true_sleep_source_1 = 64;
true_sleep_source_2 = 44;
true_sleep_source_3 = 49;
true_sleep_source_4 = 59;

novelty_weight = 10;
learning_weight = 40;
epistemic_weight = 1;
preference_weight = 10;

num_states = 100;
num_states_low = 25;

A{1}(:,:,:) = zeros(num_states,num_states,4);
a{1}(:,:,:) = zeros(num_states,num_states,4);
for i = 1:num_states
    A{1}(i,i,:) = 1;
    a{1}(i,i,:) = 1;
end
A{2}(:,:,:) = zeros(4,num_states,4);
A{2}(1,:,:) = 1;
A{2}(2,true_food_source_1,1) = 1;
A{2}(1,true_food_source_1,1) = 0;
A{2}(2,true_food_source_2,2) = 1;
A{2}(1,true_food_source_2,2) = 0;
A{2}(2,true_food_source_3,3) = 1;
A{2}(1,true_food_source_3,3) = 0;
A{2}(2,true_food_source_4,4) = 1;
A{2}(1,true_food_source_4,4) = 0;
A{2}(3,true_water_source_1,1) = 1;
A{2}(1,true_water_source_1,1) = 0;
A{2}(3,true_water_source_2,2) = 1;
A{2}(1,true_water_source_2,2) = 0;
A{2}(3,true_water_source_3,3) = 1;
A{2}(1,true_water_source_3,3) = 0;
A{2}(3,true_water_source_4,4) = 1;
A{2}(1,true_water_source_4,4) = 0;
A{2}(4,true_sleep_source_1,1) = 1;
A{2}(1,true_sleep_source_1,1) = 0;
A{2}(4,true_sleep_source_2,2) = 1;
A{2}(1,true_sleep_source_2,2) = 0;
A{2}(4,true_sleep_source_3,3) = 1;
A{2}(1,true_sleep_source_3,3) = 0;
A{2}(4,true_sleep_source_4,4) = 1;
A{2}(1,true_sleep_source_4,4) = 0;
A{3}(:,:,:) = zeros(5,num_states,4);
A{3}(5,:,:) = 1;
A{3}(1,hill_1,1) = 1;
A{3}(5,hill_1,1) = 0;
A{3}(2,hill_1,2) = 1;
A{3}(5,hill_1,2) = 0;
A{3}(3,hill_1,3) = 1;
A{3}(5,hill_1,3) = 0;
A{3}(4,hill_1,4) = 1;
A{3}(5,hill_1,4) = 0;
a{3} = A{3};
a{2}(:,:,:) = zeros(4,num_states,4);
a{2} = a{2} + 0.1; % starting concentration counts

D{1} = zeros(1,num_states)'; %position in environment
D{2} = [0.25,0.25,0.25,0.25]';

D{1}(51) = 1; % starting position
survival(:) = zeros(1,70);


D{1} = normalise(D{1});
resource_cutoffs = [24, 23, 27];
num_factors = 1;
T = 27;
num_modalities = 3;
num_iterations = 50;
TimeConst = 4;
num_states = 100;
num_states_low = 100;

short_term_memory(:,:,:,:) = zeros(35,35,35,400);

RL_state_belifs = zeros(T,num_states);
G = zeros(5);
posterior_beta = 1;
gamma(1) = 1/posterior_beta; % expected free energy precision
beta = 1;

%%% Distributions %%%


for action = 1:5
    B{1}(:,:,action)  =  eye(num_states);
    B{2}(:,:,action)  =  zeros(4);
    B{2}(:,:,action) = [0.95,   0,     0     0.05;
                        0.05,   0.95,   0,    0;
                        0,     0.05,   0.95,  0;
                        0,     0,     0.05   0.95 ];
    
    % Uniform prior over season transitions. This is what the agent must
    % learn
    b{2}(:,:,action) = [  0.25,  0.25,     0.25     0.25;
               0.25,     0.25,     0.25,    0.25;
               0.25,     0.25,     0.25,    0.25;
               0.25,     0.25,       0.25     0.25]; 
end
b = B;
for i = 1:num_states
    if i ~= [1,11,21,31,41,51,61,71,81,91]
        B{1}(:,i,2) = circshift(B{1}(:,i,2),-1); % move left
    end  
end

for i = 1:num_states
    if i ~= [10,20,30,40,50,60,70,80,90,100]
        B{1}(:,i,3) = circshift(B{1}(:,i,3),1); % move right
    end  
end

for i = 1:num_states
    if i ~= [91,92,93,94,95,96,97,98,99,100]
        B{1}(:,i,4) = circshift(B{1}(:,i,4),10); % move rup
    end  
end

for i = 1:num_states
    if i ~= [1,2,3,4,5,6,7,8,9,10]
        B{1}(:,i,5) = circshift(B{1}(:,i,5),-10); % move down
    end  
end


b{1} = B{1};
C{1} = ones(11,9); % preference for positional observation. Uniform.
C_overall{1} = zeros(T,9);


chosen_action = zeros(1,T-1);
preference_values = zeros(4,T);

for factor = 1:num_factors
    NumStates(factor) = size(B{factor},1);   % number of hidden states
    NumControllable_transitions(factor) = size(B{factor},3); % number of hidden controllable hidden states for each factor (number of B matrices)
end




time_since_food = 0;    
time_since_water = 0;
time_since_sleep = 0;
%file_name = strcat(seed,'.txt');
t = 1;

for trial = 1:120
short_term_memory(:,:,:,:) = 0;    
while(t<100 && time_since_food < 22 && time_since_water < 20 && time_since_sleep < 25)
 
    bb{2} = normalise_matrix(b{2});
    for factor = 1:2
        if t == 1
            P{t,factor} = D{factor}';
            Q{t,factor} = D{factor}';
            true_states{trial}(1, t) = 51;
            true_states{trial}(2, t) = find(cumsum(D{2}) >= rand,1);
        else
      %       P{t} = B{factor}(:,higher_level_state, higher_level_action);
            if factor == 1
                %b = B{1}(:,:,chosen_action(t-1));
                Q{t,factor} = (B{1}(:,:,chosen_action(t-1))*Q{t-1,factor}')';
                %Q{t,factor} = Q{t,factor}';
                true_states{trial}(factor, t) = find(cumsum(B{1}(:,true_states{trial}(factor,t-1),chosen_action(t-1)))>= rand,1);
            else
                %b = B{2}(:,:,:);
                Q{t,factor} = (bb{2}(:,:,chosen_action(t-1))*Q{t-1,factor}')';%(B{2}(:,:)'
                true_states{trial}(factor, t) = find(cumsum(B{2}(:,true_states{trial}(factor,t-1),1))>= rand,1);   
                 
            end
        end
        
      
    end
    
    if (true_states{trial}(2,t) == 1 && true_states{trial}(1,t) == true_food_source_1) || (true_states{trial}(2,t) == 2 && true_states{trial}(1,t) == true_food_source_2) || (true_states{trial}(2,t) == 3 && true_states{trial}(1,t) == true_food_source_3) || (true_states{trial}(2,t) == 4 && true_states{trial}(1,t) == true_food_source_4)
            time_since_food = 0;
            time_since_water = time_since_water +1;
            time_since_sleep = time_since_sleep +1;
                       
    elseif (true_states{trial}(2,t) == 1 && true_states{trial}(1,t) == true_water_source_1) || (true_states{trial}(2,t) == 2 && true_states{trial}(1,t) == true_water_source_2) || (true_states{trial}(2,t) == 3 && true_states{trial}(1,t) == true_water_source_3) || (true_states{trial}(2,t) == 4 && true_states{trial}(1,t) == true_water_source_4)
        time_since_water = 0;
        time_since_food = time_since_food +1;
        time_since_sleep = time_since_sleep +1;

    elseif (true_states{trial}(2,t) == 1 && true_states{trial}(1,t) == true_sleep_source_1) || (true_states{trial}(2,t) == 2 && true_states{trial}(1,t) == true_sleep_source_2) || (true_states{trial}(2,t) == 3 && true_states{trial}(1,t) == true_sleep_source_3) || (true_states{trial}(2,t) == 4 && true_states{trial}(1,t) == true_sleep_source_4)
        time_since_sleep = 0;
        time_since_food = time_since_food +1;
        time_since_water = time_since_water +1;
      
    else
        if t > 1
            time_since_food = time_since_food +1;
             time_since_water = time_since_water +1;
             time_since_sleep = time_since_sleep +1;
        end

    end
    % sample the next observation. Same technique as sampling states
    
    for modality = 1:num_modalities     
        ob = A{modality}(:,true_states{trial}(1,t),true_states{trial}(2,t));
        observations(modality,t) = find(cumsum(A{modality}(:,true_states{trial}(1,t),true_states{trial}(2,t)))>=rand,1);
        %create a temporary vectore of 0s
        vec = zeros(1,size(A{modality},1));
        % set the index of the vector matching the observation index to 1
        vec(1,observations(modality,t)) = 1;
        O{modality,t} = vec;
    end
    true_t = t;
    if t > 1
       
    trajectory_history = [];
    
      start = t - 6;
    if start <= 0
        start = 1;
    end
    qq = P;
    novelty = 0;
    bb{2} = normalise_matrix(b{2});
    y{2} = normalise_matrix(a{2});
    
    qs = spm_cross(Q{t,:});
    predictive_observations_posterior{2,t} = normalise(y{2}(:,:)*qs(:))'; 
    predictive_observations_posterior{3,t} = normalise(y{3}(:,:)*qs(:))';
    predicted_posterior = calculate_posterior(Q,y,predictive_observations_posterior,t);
    for timey = start:t
%          if timey ~= t
            L = spm_backwards(O,Q,A,bb,chosen_action,timey,t);
            LL{2} = L;
            LL{1} = Q{timey,1};

        if (timey > start && ~isequal(round(L,3),round(Q{timey,2},3)')) || (timey == t) 

        a_prior  = a{2};
         for modality = 2:2
           a_learning = O(modality,timey)';
           for  factor = 1:2
               a_learning = spm_cross(a_learning, LL{factor});
           end
           a_learning = a_learning.*(a{modality} > 0);
        %Define the proportion to subtract
          proportion = 0.3;
        for i = 1:size(a_learning, 3)
          % Subtract an amount proportional to the maximum value from each zero entry in each column
           for j = 1:size(a_learning, 2)
               max_value = max(a_learning(2:end,j,i)); % find the maximum value in column j
               amount_to_subtract = proportion * max_value; % calculate the amount to subtract
               a_learning(a_learning(1,j,i) ==0,j ,i) = a_learning(a_learning(1,j,i) ==0,j ,i) - amount_to_subtract;
               
           end
        end
        a{modality} = a{modality} + 0.7*a_learning;
        a{modality}(a{modality} <=0.05) = 0.05;
        
         end
        a1 =a{2};
        a1 = a1(:);
         
        a2 = a_prior;
        a2 = a2(:);
         
         w = kldir(normalise(a2(:)),normalise(a1(:)));
         
        end
    end
    end
     
      if true_states{trial}(2,t) == 1
           food = true_food_source_1;
           water = true_water_source_1;
           sleep = true_sleep_source_1;
      elseif true_states{trial}(2,t) == 2
          food = true_food_source_2;
           water = true_water_source_2;
           sleep = true_sleep_source_2;
      elseif true_states{trial}(2,t) == 3
          food = true_food_source_3;
           water = true_water_source_3;
           sleep = true_sleep_source_3;
      else
          food = true_food_source_4;
           water = true_water_source_4;
           sleep = true_sleep_source_4;
      end

    y{2} = normalise_matrix(a{2});
    y{1} = A{1};
    y{3} = A{3};
    displayGridWorld(true_states{trial}(1,t),food,water,sleep, hill_1, 1)
    horizon = min([9,min([22-time_since_food, 20-time_since_water, 25-time_since_sleep])]);
    if horizon == 0
        horizon = 1;
    end
    temp_Q = Q;
    temp_Q{t,2} = temp_Q{t,2}';
    P = calculate_posterior(temp_Q,y,O,t);
    current_pos = find(cumsum(P{t,1})>=rand,1);
    if t > 1 && ~isequal(round(predicted_posterior{t,2},1), round(P{t,2},1))
        % if there is a relatively large state-prediction error, reset
        % memory as it's probably innacurate. 
          short_term_memory(:,:,:,:) = 0;      
    end
    if current_pos == 55
        short_term_memory(:,:,:,:) = 0;
    end

    cur_state = spm_cross(P{t});
    cur_state = find(cumsum(cur_state(:))>=rand,1);
    best_actions = [];
    % Start tree search from current time point
   [G,Q, short_term_memory, best_actions] = tree_search_frwd_SI(short_term_memory, O, Q ,a, A,y, B,B, t, T, t+horizon, time_since_food, time_since_water, time_since_sleep, true_t, chosen_action, time_since_food, time_since_water, time_since_sleep, best_actions, learning_weight, novelty_weight, epistemic_weight, preference_weight);
        
   chosen_action(t) = best_actions(1);
   t = t+1;
    % end loop over time points

end

fid = fopen(file_name, 'a+');
fprintf(fid, '%f\n', t);

% reset for next iteration
t = 1;
time_since_food = 0;
time_since_water = 0;
time_since_sleep = 0;
end

%end




















%%%%%%%%%%%% code for graphical depiction of simulations %%%%%%%%%%%%%

% function a = displayGridWorld(agent_position, food_position_1,water_position_1,sleep_position_1,hill_1_pos,alive_status)
% if alive_status == 1
%     agent_text = 'A';
% else 
%     agent_text = 'Dead';
% end
% 
% agent_dim1 = 0;
% if agent_position <= 10
%     agent_dim2 = 1;
%     agent_dim1 = agent_position;
% elseif agent_position < 21
%     agent_dim2 = 2;
%     agent_dim1 = agent_position - 10;
% elseif agent_position < 31
%     agent_dim2 = 3;
%     agent_dim1 = agent_position - 20;
% elseif agent_position < 41
%     agent_dim2 = 4;
%     agent_dim1 = agent_position - 30;
% elseif agent_position < 51
%     agent_dim2 = 5;
%     agent_dim1 = agent_position - 40;
% elseif agent_position < 61
%     agent_dim2 = 6;
%     agent_dim1 = agent_position - 50;
% elseif agent_position < 71
%     agent_dim2 = 7;
%     agent_dim1 = agent_position - 60;
% elseif agent_position < 81
%     agent_dim2 = 8;
%     agent_dim1 = agent_position - 70;
% elseif agent_position < 91
%     agent_dim2 = 9;
%     agent_dim1 = agent_position - 80;
% else
%     agent_dim2 = 10;
%     agent_dim1 = agent_position - 90;
% end
% 
% locations_1 = [];
% hill_1_dim2 = idivide(int16(hill_1_pos),10,'floor')+1;
% hill_1_dim1 = rem(hill_1_pos,10);
% if hill_1_dim1 == 0
%     if hill_1_dim2 ~= 1
%         hill_1_dim2 = hill_1_dim2-1;
%     end
%     hill_1_dim1 = 10;
% end
% 
% food_1_dim2 = idivide(int16(food_position_1),10,'floor')+1;
% food_1_dim1 = rem(food_position_1,10);
% if food_1_dim1 == 0
%     if food_1_dim2 ~= 1
%         food_1_dim2 = food_1_dim2-1;
%     end
%     food_1_dim1 = 10;
% end
% locations_1(end+1) = food_1_dim1;
% % food_2_dim2 = idivide(int16(food_position_2), 10, 'floor')+1;
% % food_2_dim1 = rem(food_position_2,10);
% % if food_2_dim1 == 0
% %     food_2_dim1 = 10;
% %     if food_2_dim2 ~= 1
% %         food_2_dim2 = food_2_dim2-1;
% %     end
% % end
% % locations_1(end+1) = food_2_dim1;
% % food_3_dim2 = idivide(int16(food_position_3), 10, 'floor')+1;
% % food_3_dim1 = rem(food_position_3, 10);
% % if food_3_dim1 == 0
% %     food_3_dim1 = 10;
% %     if food_3_dim2 ~= 1
% %         food_3_dim2 = food_3_dim2-1;
% %     end
% % end
% % locations_1(end+1) = food_3_dim1;
% % 
% % food_4_dim2 = idivide(int16(food_position_4), 10, 'floor')+1;
% % food_4_dim1 = rem(food_position_4, 10);
% % if food_4_dim1 == 0
% %     food_4_dim1 = 10;
% %     if food_4_dim2 ~= 1
% %         food_4_dim2 = food_4_dim2-1;
% %     end
% % end
% % locations_1(end+1) = food_4_dim1;
% 
% water_1_dim2 = idivide(int16(water_position_1), 10, 'floor')+1;
% water_1_dim1 = rem(water_position_1, 10);
% if water_1_dim1 == 0
%     water_1_dim1 = 10;
%     if water_1_dim2 ~= 1
%         water_1_dim2 = water_1_dim2-1;
%     end
% end
% 
% % water_2_dim2 = idivide(int16(water_position_2), 10, 'floor')+1;
% % water_2_dim1 = rem(water_position_2, 10);
% % if water_2_dim1 == 0
% %     water_2_dim1 = 10;
% %     if water_2_dim2 ~= 1
% %         water_2_dim2 = water_2_dim2 - 1;
% %     end
% % end
% % locations_1(end+1) = water_2_dim1;
% % 
% % water_3_dim2 = idivide(int16(water_position_3), 10, 'floor')+1;
% % water_3_dim1 = rem(water_position_3, 10);
% % if water_3_dim1 == 0
% %     water_3_dim1 = 10;
% %     if water_3_dim2 ~= 1
% %         water_3_dim2 = water_3_dim2 - 1;
% %     end
% % end
% % locations_1(end+1) = water_2_dim1;
% % 
% % water_4_dim2 = idivide(int16(water_position_4), 10, 'floor')+1;
% % water_4_dim1 = rem(water_position_4, 10);
% % if water_4_dim1 == 0
% %     water_4_dim1 = 10;
% %     if water_4_dim2 ~= 1
% %         water_4_dim2 = water_4_dim2 - 1;
% %     end
% % end
% % locations_1(end+1) = water_2_dim1;
% sleep_1_dim2 = idivide(int16(sleep_position_1), 10, 'floor')+1;
% sleep_1_dim1 = rem(sleep_position_1, 10);
% if sleep_1_dim1 == 0
%     sleep_1_dim1 = 10;
%     if sleep_1_dim2 ~= 1
%         sleep_1_dim2 = sleep_1_dim2-1;
%     end
% end
% locations_1(end+1) = sleep_1_dim1;
% % sleep_2_dim2 = idivide(int16(sleep_position_2), 10, 'floor')+1;
% % sleep_2_dim1 = rem(sleep_position_2, 10);
% % if sleep_2_dim1 == 0
% %     sleep_2_dim1 = 10;
% %     if sleep_2_dim2 ~= 1
% %         sleep_2_dim2 = sleep_2_dim2-1;
% %     end
% % end
% % 
% % sleep_3_dim2 = idivide(int16(sleep_position_3), 10, 'floor')+1;
% % sleep_3_dim1 = rem(sleep_position_2, 10);
% % if sleep_3_dim1 == 0
% %     sleep_3_dim1 = 10;
% %     if sleep_3_dim2 ~= 1
% %         sleep_3_dim2 = sleep_3_dim2-1;
% %     end
% % end
% % 
% % sleep_4_dim2 = idivide(int16(sleep_position_4), 10, 'floor')+1;
% % sleep_4_dim1 = rem(sleep_position_4, 10);
% % if sleep_4_dim1 == 0
% %     sleep_4_dim1 = 10;
% %     if sleep_4_dim2 ~= 1
% %         sleep_4_dim2 = sleep_4_dim2-1;
% %     end
% 
% 
% 
% h1=figure(1);
% set(h1,'name','gridworld');
% h1.Position = [400 200 800 700];
% [X,Y]=meshgrid(1:11,1:11);
% plot(Y,X,'k'); hold on; axis off
% plot(X,Y,'k');hold off; axis off
% hold off;
% I=(1);
% surface(I);
% h=linspace(0.5,1,64);
% %h=[h',h',h'];
% %set(gcf,'Colormap',h);
% q=1;
% x=linspace(1.5,10.5,10);
% y=linspace(1.5,10.5,10);
% %empty_pref =sprintf('%.3f',preference_values(1));
% %food_pref =sprintf('%.3f',preference_values(2));
% %water_pref =sprintf('%.3f',preference_values(3));
% %sleep_pref =sprintf('%.3f',preference_values(4));
% for n=1:10
%     for p=1:10
%         if n == agent_dim1 & p == agent_dim2
%             text(y(n)-.2,x(p),agent_text,'FontSize',16);
%             q=q+1;
% 
%         end
% 
%         if (n == food_1_dim1 & p == food_1_dim2) 
%             text(y(n)-.2,x(p)+.3,'F','FontSize',16, 'FontWeight','bold');
%             %text(y(n)-.2,x(p)-.3,food_pref,'FontSize', 12);
%             q=q+1;
%         end
% 
% %          if (n == food_2_dim1 & p == food_2_dim2) 
% %             text(y(n)-.2,x(p)+.3,'F','FontSize',16, 'FontWeight','bold');
% %             %text(y(n)-.2,x(p)-.3,food_pref,'FontSize', 12);
% %             q=q+1;
% %          end
% % 
% %           if (n == food_3_dim1 & p == food_3_dim2) 
% %             text(y(n)-.2,x(p)+.3,'F','FontSize',16, 'FontWeight','bold');
% %             %text(y(n)-.2,x(p)-.3,food_pref,'FontSize', 12);
% %             q=q+1;
% %           end
% % 
% %            if (n == food_4_dim1 & p == food_4_dim2) 
% %             text(y(n)-.2,x(p)+.3,'F','FontSize',16, 'FontWeight','bold');
% %             %text(y(n)-.2,x(p)-.3,food_pref,'FontSize', 12);
% %             q=q+1;
% %         end
% 
%         if (n == water_1_dim1 & p == water_1_dim2) 
%             text(y(n)-.2,x(p)+.3,'W','FontSize',16, 'FontWeight','bold');
%             %text(y(n)-.2,x(p)-.3,water_pref,'FontSize', 12);
%             q=q+1;
%         end
% 
% %          if (n == water_2_dim1 & p == water_2_dim2) 
% %             text(y(n)-.2,x(p)+.3,'W','FontSize',16, 'FontWeight','bold');
% %             %text(y(n)-.2,x(p)-.3,water_pref,'FontSize', 12);
% %             q=q+1;
% %          end
% % 
% %           if (n == water_3_dim1 & p == water_3_dim2) 
% %             text(y(n)-.2,x(p)+.3,'W','FontSize',16, 'FontWeight','bold');
% %             %text(y(n)-.2,x(p)-.3,water_pref,'FontSize', 12);
% %             q=q+1;
% %           end
% % 
% %            if (n == water_4_dim1 & p == water_4_dim2) 
% %             text(y(n)-.2,x(p)+.3,'W','FontSize',16, 'FontWeight','bold');
% %             %text(y(n)-.2,x(p)-.3,water_pref,'FontSize', 12);
% %             q=q+1;
% %         end
% 
% 
%         if (n == hill_1_dim1 & p == hill_1_dim2)
%             text(y(n)-.2,x(p)+.3,'Hill','FontSize',16, 'FontWeight','bold');
%             %text(y(n)-.2,x(p)-.3,water_pref,'FontSize', 12);
%             q=q+1;
%         end
% 
% 
%         if (n == sleep_1_dim1 & p == sleep_1_dim2)
%             text(y(n)-.2,x(p)+.3,'S','FontSize',16, 'FontWeight','bold');
%             %text(y(n)-.2,x(p)-.3,sleep_pref,'FontSize', 12);
%             q=q+1;
%         end
% 
% %         if (n == sleep_3_dim1 & p == sleep_3_dim2)
% %             text(y(n)-.2,x(p)+.3,'S','FontSize',16, 'FontWeight','bold');
% %             %text(y(n)-.2,x(p)-.3,sleep_pref,'FontSize', 12);
% %             q=q+1;
% %         end
% % 
% %         if (n == sleep_3_dim1 & p == sleep_3_dim2)
% %             text(y(n)-.2,x(p)+.3,'S','FontSize',16, 'FontWeight','bold');
% %             %text(y(n)-.2,x(p)-.3,sleep_pref,'FontSize', 12);
% %             q=q+1;
% %         end
% % 
% %         if (n == sleep_4_dim1 & p == sleep_4_dim2)
% %             text(y(n)-.2,x(p)+.3,'S','FontSize',16, 'FontWeight','bold');
% %             %text(y(n)-.2,x(p)-.3,sleep_pref,'FontSize', 12);
% %             q=q+1;
% %         end
% 
%         %if ~(n == sleep_dim1 && p == sleep_dim2) && ~(n == food_1_dim1 && p == food_1_dim2) && ~(n == food_2_dim1 && p == food_2_dim2) && ~(n == food_3_dim1 && p == food_3_dim2) && ~(n == water_1_dim1 && p == water_1_dim2) && ~(n ==water_2_dim1 && p == water_2_dim2)
%          %  text(y(n)-.2,x(p)-.3,empty_pref,'FontSize', 12);
%         %end
% 
%     end
% end
% 
% 
% %pause(0.5)
% 
% end


%--------------------------------------------------------------------------















%function kl = kldir(a,b)
%kl = sum(a.*(log(a)-log(b)),'all');
%end





        
        
   



   
