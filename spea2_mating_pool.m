function [Popul] = spea2_mating_pool(MutMax,CrosMax,ExtSet,ExtSize,PopSize,ExtSet_Fit,iter,GenNumb,ell,featNum)

Popul_aux1 = zeros(PopSize,ell);
MP1 = randi([1,ExtSize],[1, PopSize]);
MP2 = randi([1,ExtSize],[1, PopSize]);
[~, ExtSet_Fit_sort_ind] = sort(ExtSet_Fit);

for ii=1:PopSize
    if MP1(ii) <= MP2(ii)
        Popul_aux1(ii,:) = ExtSet(ExtSet_Fit_sort_ind(MP1(ii)),:);
    else
        Popul_aux1(ii,:) = ExtSet(ExtSet_Fit_sort_ind(MP2(ii)),:);
    end
end

% Crossover / Recombination
Popul_aux2 = zeros(PopSize,ell);
Cross_aux_perm = randperm(CrosMax);
for ii=1:round(CrosMax/2)
    inter = intersect(Popul_aux1(Cross_aux_perm(2),:),Popul_aux1(Cross_aux_perm(1),:));
    
    if length(inter) >= (ell-1)
        Popul_aux2(2*ii-1,:) = Popul_aux1(Cross_aux_perm(1),:);
        Popul_aux2(2*ii,:) = Popul_aux1(Cross_aux_perm(2),:);
    else
        Popul_aux2(2*ii-1,1:length(inter)) = inter;
        Popul_aux2(2*ii,1:length(inter)) = inter;
        P1_aux = setdiff(Popul_aux1(Cross_aux_perm(1),:),inter);
        P2_aux = setdiff(Popul_aux1(Cross_aux_perm(2),:),inter);
        prob = randperm(length(P1_aux));
        prob2 = randi([1,round((iter-1)*(1-(length(P1_aux)-1))/(GenNumb-1)+(length(P1_aux)-1))],1);
        Popul_aux2(2*ii-1,length(inter)+1:length(inter)+prob2) = P2_aux(1,prob(1:prob2));
        Popul_aux2(2*ii-1,length(inter)+prob2+1:end) = P1_aux(1,prob(prob2+1:end));
        Popul_aux2(2*ii,length(inter)+1:length(inter)+prob2) = P1_aux(1,prob(1:prob2));
        Popul_aux2(2*ii,length(inter)+prob2+1:end) = P2_aux(1,prob(prob2+1:end));
    end
    
    Cross_aux_perm(1:2) = [];
end

% Mutation
Mut_aux_perm = randperm(MutMax)+CrosMax;
for ii=round(CrosMax/2)*2+1:PopSize
    prob = randperm(ell);
    %     prob2 = randi([1,3],1);
    prob2 = randi([1,round((iter-1)*(1-(ell))/(GenNumb-1)+(ell))],1);
    P_aux = setdiff(1:featNum,[Popul_aux1(Mut_aux_perm(1),prob(prob2+1:end))]);
    prob3 = randperm(length(P_aux));
    Popul_aux2(ii,prob(1:prob2)) = P_aux(1,prob3(1:prob2));
    Popul_aux2(ii,prob(prob2+1:end)) = Popul_aux1(Mut_aux_perm(1),prob(prob2+1:end));
    Mut_aux_perm(1) = [];
end

Popul = Popul_aux2;

end

