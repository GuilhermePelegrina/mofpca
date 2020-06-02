function [PopExt_Fit] = spea2_fitness(PopExt_Size,PopExt_Cost,ParK)

% Strength
PopExt_Sm = zeros(PopExt_Size,PopExt_Size);
PopExt_S = zeros(PopExt_Size,1);
for ii=1:PopExt_Size
    for jj=ii:PopExt_Size
        if all(PopExt_Cost(ii,:) <= PopExt_Cost(jj,:)) && any(PopExt_Cost(ii,:) < PopExt_Cost(jj,:))
            PopExt_Sm(ii,jj) = 1;
        elseif all(PopExt_Cost(jj,:) <= PopExt_Cost(ii,:)) && any(PopExt_Cost(jj,:) < PopExt_Cost(ii,:))
            PopExt_Sm(jj,ii) = 1;
        end
    end
    PopExt_S(ii) = sum(PopExt_Sm(ii,:)); % Strength
end

% Raw fitness
Raw_aux = repmat(PopExt_S,1,PopExt_Size);
Raw_aux2 = Raw_aux.*PopExt_Sm;
PopExt_R = sum(Raw_aux2)'; % Raw fitness
    
f1_ideal = min(PopExt_Cost(:,1)); f2_ideal = min(PopExt_Cost(:,2)); % Solução ideal para C1
f1_nadir = max(PopExt_Cost(:,1)); f2_nadir = max(PopExt_Cost(:,2)); % Solução nadir para C2
f1_norm = (PopExt_Cost(:,1) - f1_ideal)/(f1_nadir - f1_ideal);             % Normalizando f1 para C1
f2_norm = (PopExt_Cost(:,2) - f2_ideal)/(f2_nadir - f2_ideal);             % Normalizando f2 para C2
PopExt_DistVec = pdist([f1_norm, f2_norm]);
PopExt_DistMatr = squareform(PopExt_DistVec); % Distance matrix
[PopExt_DistMatr, ~] = sort(PopExt_DistMatr'); % Ordering and keeping the index
PopExt_DistMatr(1,:) = [];

% Density
PopExt_Dens = [];
PopExt_Dens(:,1) = 1./(PopExt_DistMatr(ParK,1:PopExt_Size) + 2);

% Fitness
PopExt_Fit = PopExt_R + PopExt_Dens;

end

