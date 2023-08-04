function C = determineObservationPreference(t_food, t_water, t_sleep)

empty = -1 ;

if t_water > 19
   t_food = -500;
   t_sleep = -500;
   empty = -500;    
end

if t_food > 21
   t_water = -500;
   t_sleep = -500;
   empty = -500;

end
if t_sleep >24
   t_food = -500;
   t_water = -500;
   empty = -500;
end

C{2} = [empty, t_food, t_water, t_sleep];

end
