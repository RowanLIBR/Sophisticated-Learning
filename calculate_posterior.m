function P = calculate_posterior(P,A,O,t)
if size(P{t,2},2) > 1
    P{t,2} = P{t,2}';
end
for fact = 2:2
        L     = 1;
        num = numel(A);
        for modal = 2:num
                obs = find(cumsum(O{modal,t})>= rand,1);
                temp = A{modal}(obs,:,:);
                temp = permute(temp,[3,2,1]);
                L = L.*temp;
        end
        %L = permute(L,[3,2,1]);
        for f = 1:2
            if f ~= fact
                if f == 2
                    LL = P{t,f}*L;
                else
                    LL = L*P{t,f}';
                end
            end
        end
        y = LL.*P{t,fact};
        P{t,fact}  = normalise(y)';       
end
end
