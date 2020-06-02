function [ExtSet,ExtSet_Cost,ExtSet_Fit] = spea2_selection(ExtSize,PopExt_Size,PopExt_Fit,PopExt,PopExt_Cost)

ExtSet = [];
ExtSet_Cost = [];
ExtSet_Fit = [];
for ii=1:PopExt_Size
    if PopExt_Fit(ii) < 1
        ExtSet = [ExtSet; PopExt(ii,:)]; % Updating external set
        ExtSet_Cost = [ExtSet_Cost; PopExt_Cost(ii,:)]; % Updating external set cost function
        ExtSet_Fit = [ExtSet_Fit; PopExt_Fit(ii)];
    end
end

f1_ideal = min(ExtSet_Cost(:,1)); f2_ideal = min(ExtSet_Cost(:,2)); % Solução ideal para C1
f1_nadir = max(ExtSet_Cost(:,1)); f2_nadir = max(ExtSet_Cost(:,2)); % Solução nadir para C2
f1_norm = (ExtSet_Cost(:,1) - f1_ideal)/(f1_nadir - f1_ideal);             % Normalizando f1 para C1
f2_norm = (ExtSet_Cost(:,2) - f2_ideal)/(f2_nadir - f2_ideal);             % Normalizando f2 para C2
ExtSet_DistVec = pdist([f1_norm, f2_norm]);
ExtSet_DistMatr = squareform(ExtSet_DistVec); % Distance matrix
ExtSet_DistMatr = ExtSet_DistMatr - eye(length(f1_norm)); % Diagonal receives (-1)

% Analysis of length of external set
if size(ExtSet,1) > ExtSize % Case with more individuals than the maximum
    while size(ExtSet,1) > ExtSize
        
        [ExtSet_DistMatr2, ExtSet_DistMatr_ind] = sort(ExtSet_DistMatr'); % Ordering and keeping the index
        ExtSet_DistMatr2(1,:) = [];
        ExtSet_DistMatr_ind(1,:) = [];
        
        [~, ExtSet_min_ind] = min(ExtSet_DistMatr2(1,:));
        if ExtSet_DistMatr2(2,ExtSet_min_ind) < ExtSet_DistMatr2(2,ExtSet_DistMatr_ind(1,ExtSet_min_ind))
            ExtSet(ExtSet_min_ind,:) = [];
            ExtSet_Cost(ExtSet_min_ind,:) = [];
            ExtSet_Fit(ExtSet_min_ind) = [];
            ExtSet_DistMatr(ExtSet_min_ind,:) = [];
            ExtSet_DistMatr(:,ExtSet_min_ind) = [];
        else 
            ExtSet(ExtSet_DistMatr_ind(1,ExtSet_min_ind),:) = [];
            ExtSet_Cost(ExtSet_DistMatr_ind(1,ExtSet_min_ind),:) = [];
            ExtSet_Fit(ExtSet_DistMatr_ind(1,ExtSet_min_ind)) = [];
            ExtSet_DistMatr(ExtSet_DistMatr_ind(1,ExtSet_min_ind),:) = [];
            ExtSet_DistMatr(:,ExtSet_DistMatr_ind(1,ExtSet_min_ind)) = [];
        end
    end
elseif size(ExtSet,1) < ExtSize % Case with less individuals than the maximum
    [~, PopExt_Fit_sort_ind] = sort(PopExt_Fit);
    ExtSet_Cost = [ExtSet_Cost; PopExt_Cost(PopExt_Fit_sort_ind(size(ExtSet,1)+1:ExtSize),:)];
    ExtSet_Fit = [ExtSet_Fit; PopExt_Fit(PopExt_Fit_sort_ind(size(ExtSet,1)+1:ExtSize))];
    ExtSet = [ExtSet; PopExt(PopExt_Fit_sort_ind(size(ExtSet,1)+1:ExtSize),:)];
end


end
