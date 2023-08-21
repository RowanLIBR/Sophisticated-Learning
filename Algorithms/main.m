run_SL = 1;
run_SI = 0;
run_BA = 0;
run_BAUCB = 0;

if run_SL == 1
    SL("1")
elseif run_SI == 1
    SI("1")
elseif run_BA == 1
    BA("1")
elseif run_BAUCB == 1
    BA_UCB("1")
end