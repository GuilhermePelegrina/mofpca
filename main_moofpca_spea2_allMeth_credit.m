%% This script verifies the compromise between the (total) reconstruction error and the fairnes measure in a dimensional reduction problem
% Fairness measure is given by the difference between the reconstruction errors of the two considered classes.
% We use a multi-objective approach (SPEA2 algorithm) and select a single non-dominated solution based on
% the minimum weighted sum (with equal importance, but taking the scales of each objective into account).
% We consider the Default Credit dataset - see: [Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining
% techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with
% Applications, 36(2), p. 2473-2480] - and some functions were based on [Samadi et al. (2018). The price of fair pca:
% One extra dimension. In Advances in Neural Information Processing Systems, p. 10976-10987].

% To cite this work: Pelegrina, G. D.; Brotto, R. D. B.; Duarte, L. T.; Attux, R. & Romano, J. M. T. (2020).
% A multi-objective-based approach for Fair Principal Component Analysis. ArXiv preprint, arXiv:2006.06137.
% Available at: https://arxiv.org/abs/2006.06137

clc; clear all; close all;

%% Input data and parameters
[M, ~, ~, A, B] = creditProcess_mono(); % Load data (education as the sensitive variable)
featNum = size(A,2); % Number of features
featNumUse = featNum-1; % Maximum number of feature to be considered in dimensional reduction

% [M, ~, ~, A, B] = LFWProcess_mono(); % Load data (gender as the sensitive variable)
% featNum = size(A,2); % Number of features
% featNumUse = 20; % Maximum number of feature to be considered in dimensional reduction

%% Non-supervised PCA for all classes
coeff = pca(M); % Eigenvectors coefficients

%% Multiobjective Optimization - SPEA2
% Inputs
GenNumb = 50; % Maximum number of generation
EvalFuncs=@(theta) spea2_CostFuncs(theta,coeff,A,B); % Cost functions
CrosPro = 0.5; % Crossover probability
MutRate = 1 - CrosPro; % Mutation rate

best_cost1 = zeros(featNumUse,2); % Matrix that stores the cost functions for the selected non-dominated solution
best_coeff1 = zeros(featNumUse,featNumUse); % Matrix that stores the coefficients of the selected non-dominated solution

% For one feature - brute force
for ii=1:featNum
    Costs(ii,:) = EvalFuncs(ii);
end
include = 1:featNum; % Indices of non-dominated solutions of cardinality |featureNum|
for kk=1:2
    ind1 = 1; % Auxiliar variable
    while ind1 < length(include)
        include([zeros(ind1-1,1)==1;all((repmat(Costs(include(ind1),:),length(include(ind1:end)),1) < Costs(include(ind1:end),:))')'])=[];
        ind1 = ind1 + 1;
    end
    include = flip(include);
end

pareto_cost = Costs(include,:); % Cost functions of the non-dominated solutions
pareto_coeff = [include', zeros(size(pareto_cost,1),featNumUse-1)];  % Coefficients of the non-dominated solutions
pareto_numb(1:2,1) = [1;size(pareto_cost,1)+1]; % Auxiliar vector that stores the solution indices for different number of features

if size(pareto_cost,1) ~= 1 % For coalitions with more than one non-dominated solutions
    % Weights used to select a single non-dominated solution
    a=max(pareto_cost(:,1))-min(pareto_cost(:,1));
    b=max(pareto_cost(:,2))-min(pareto_cost(:,2));
    py = a/(a+b); % Weights for the second cost function
    px = b/(a+b); % Weights for the first cost function
    
    % Method 1 - Weighted sum based on the initial scales
    weightSum = [px, py]*pareto_cost'; % Calculate the Euclidian distances
    [order_weight,index_weight] = sort(weightSum); % Ordering
    % If two solutions achieves the same value, we chose the one that minimizes the first cost function (reconstruction error)
    if abs(order_weight(1) - order_weight(2)) < eps
        [order_aux,order_aux_ind] = min([pareto_cost(index_weight(1),1),pareto_cost(index_weight(2),1)]);
        best_cost1(1,:) = pareto_cost(index_weight(order_aux_ind),:);
        best_coeff1(1,:) = [include(index_weight(order_aux_ind)), zeros(1,featNumUse-1)];
    else
        best_cost1(1,:) = pareto_cost(index_weight(1),:);
        best_coeff1(1,:) = [include(index_weight(1)), zeros(1,featNumUse-1)];
    end
    
    % Method 2 - Weighted sum (0.5/0.5) based on actual values
    minim = min(pareto_cost);
    maxim = max(pareto_cost);
    par_aux = [];
    par_aux(:,1) = (pareto_cost(:,1)-minim(1))/(maxim(1)-minim(1));
    par_aux(:,2) = (pareto_cost(:,2)-minim(2))/(maxim(2)-minim(2));
    
    weightSum = [0.5, 0.5]*par_aux'; % Calculate the Euclidian distances
    [order_weight,index_weight] = sort(weightSum); % Ordering
    if abs(order_weight(1) - order_weight(2)) < eps
        [order_aux,order_aux_ind] = min([pareto_cost(index_weight(1),1),pareto_cost(index_weight(2),1)]);
        best_cost2(1,:) = pareto_cost(index_weight(order_aux_ind),:);
        best_coeff2(1,:) = [include(index_weight(order_aux_ind)), zeros(1,featNumUse-1)];
    else
        best_cost2(1,:) = pareto_cost(index_weight(1),:);
        best_coeff2(1,:) = [include(index_weight(1)), zeros(1,featNumUse-1)];
    end
    
    % Method 3 - Best reconstruction error
    [order_aux,order_aux_ind] = min(pareto_cost(:,1));
    best_cost3(1,:) = pareto_cost(order_aux_ind,:);
    best_coeff3(1,:) = [include(order_aux_ind), zeros(1,featNumUse-1)];
    
    % Method 4 - Best fairness measure
    [order_aux,order_aux_ind] = min(pareto_cost(:,2));
    best_cost4(1,:) = pareto_cost(order_aux_ind,:);
    best_coeff4(1,:) = [include(order_aux_ind), zeros(1,featNumUse-1)];
else
    py = pareto_cost(1)/sum(pareto_cost); % Weights for the second cost function
    px = pareto_cost(2)/sum(pareto_cost); % Weights for the first cost function
    best_cost1(1,:) = pareto_cost;
    best_cost2(1,:) = pareto_cost;
    best_cost3(1,:) = pareto_cost;
    best_cost4(1,:) = pareto_cost;
    best_coeff1(1,:) = [include, zeros(1,featNumUse-1)];
    best_coeff2(1,:) = [include, zeros(1,featNumUse-1)];
    best_coeff3(1,:) = [include, zeros(1,featNumUse-1)];
    best_coeff4(1,:) = [include, zeros(1,featNumUse-1)];
end

% For more than one feature - MOFPCA
for ell=2:featNumUse
    % MO parameters
    PopSize = min(100,round(nchoosek(featNum,ell)/2)); % Population size
    ExtSize = round(PopSize/2); % Maximum external set size
    ParK = round(sqrt(PopSize + ExtSize)); % SPEA2 parameter
    CrosMax = round(CrosPro*PopSize/2)*2; % Maximum number of crossover
    MutMax = PopSize - CrosMax; % Maximum number of mutation
    
    %% Inicialization
    Popul = [];
    ii = 1;
    while ii <= PopSize % Population with different elements
        aux_Pop = sort(randperm(featNum,ell));
        if size(unique([Popul;aux_Pop],'rows'),1) == ii
            Popul = [Popul; aux_Pop]; % Initial population
            ii = ii + 1;
        end
    end
    Popul(1,:) = 1:ell;
    
    ExtSet = []; % External set (empty at the beginning)
    ExtSet_Cost = []; % External set cost (empty at the beginning)
    evol = 0; % Counter for non-dominated solutions evolution
    
    for iter = 1:GenNumb
        
        PopExt = [Popul; ExtSet]; % Union of population and external set
        PopExt_Size = size(PopExt,1);
        PopExt_Cost = zeros(PopSize,2);
        PopExt_Cost(1,:) = EvalFuncs(PopExt(1,:)); % Evaluation of cost functions for population + external set
        for ii=2:PopSize
            if all(PopExt(ii,:)==PopExt(ii-1,:))
                PopExt_Cost(ii,:) = PopExt_Cost(ii-1,:); % Evaluation of cost functions for population + external set
            else
                PopExt_Cost(ii,:) = EvalFuncs(PopExt(ii,:)); % Evaluation of cost functions for population + external set
            end
        end
        PopExt_Cost = [PopExt_Cost;ExtSet_Cost];
        
        % Fitness
        [PopExt_Fit] = spea2_fitness(PopExt_Size,PopExt_Cost,ParK);
        
        % Selection
        [ExtSet,ExtSet_Cost,ExtSet_Fit] = spea2_selection(ExtSize,PopExt_Size,PopExt_Fit,PopExt,PopExt_Cost);
        
        % Stop criteria
        if iter >= GenNumb
            break;
        end
        
        % Mating pool
        [Popul] = spea2_mating_pool(MutMax,CrosMax,ExtSet,ExtSize,PopSize,ExtSet_Fit,iter,GenNumb,ell,featNum);
        Popul = sortrows(Popul);
        
        [iter, ell]
    end
    
    % Updating the non-dominated solutions
    [pareto_aux,pareto_aux_ind,~] = unique(ExtSet_Cost,'rows');
    ExtSet_pareto = ExtSet(pareto_aux_ind,:);
    pareto_coeff = [pareto_coeff; [ExtSet_pareto, zeros(size(pareto_aux,1),featNumUse-size(ExtSet,2))]];
    pareto_cost = [pareto_cost; pareto_aux];
    pareto_numb(ell+1) = pareto_numb(ell) + size(pareto_aux,1);
    
    % Selecting a single non-dominated solution
    if size(pareto_aux,1) ~= 1 % For coalitions with more than one non-dominated solutions
        
        % Method 1 - Weighted sum based on the initial scales
        weightSum = [px, py]*pareto_aux'; % Calculate the Euclidian distances
        [order_weight,index_weight] = sort(weightSum); % Ordering
        if abs(order_weight(1) - order_weight(2)) < eps
            [order_aux,order_aux_ind] = min([pareto_aux(index_weight(1),1),pareto_aux(index_weight(2),1)]);
            best_cost1(ell,:) = pareto_aux(index_weight(order_aux_ind),:);
            best_coeff1(ell,:) = [ExtSet_pareto(index_weight(order_aux_ind),:), zeros(1,featNumUse-size(ExtSet,2))];
        else
            best_cost1(ell,:) = pareto_aux(index_weight(1),:);
            best_coeff1(ell,:) = [ExtSet_pareto(index_weight(1),:), zeros(1,featNumUse-size(ExtSet,2))];
        end
        
        % Method 2 - Weighted sum (0.5/0.5) based on actual values
        minim = min(pareto_aux);
        maxim = max(pareto_aux);
        par_aux = [];
        par_aux(:,1) = (pareto_aux(:,1)-minim(1))/(maxim(1)-minim(1));
        par_aux(:,2) = (pareto_aux(:,2)-minim(2))/(maxim(2)-minim(2));
        
        weightSum = [0.5, 0.5]*par_aux'; % Calculate the Euclidian distances
        [order_weight,index_weight] = sort(weightSum); % Ordering
        if abs(order_weight(1) - order_weight(2)) < eps
            [order_aux,order_aux_ind] = min([pareto_aux(index_weight(1),1),pareto_aux(index_weight(2),1)]);
            best_cost2(ell,:) = pareto_aux(index_weight(order_aux_ind),:);
            best_coeff2(ell,:) = [ExtSet_pareto(index_weight(order_aux_ind),:), zeros(1,featNumUse-size(ExtSet,2))];
        else
            best_cost2(ell,:) = pareto_aux(index_weight(1),:);
            best_coeff2(ell,:) = [ExtSet_pareto(index_weight(1),:), zeros(1,featNumUse-size(ExtSet,2))];
        end
        
        % Method 3 - Best reconstruction error
        [order_aux,order_aux_ind] = min(pareto_aux(:,1));
        best_cost3(ell,:) = pareto_aux(order_aux_ind,:);
        best_coeff3(ell,:) = [ExtSet_pareto(order_aux_ind,:), zeros(1,featNumUse-size(ExtSet,2))];
        
        % Method 4 - Best fairness measure
        [order_aux,order_aux_ind] = min(pareto_aux(:,2));
        best_cost4(ell,:) = pareto_aux(order_aux_ind,:);
        best_coeff4(ell,:) = [ExtSet_pareto(order_aux_ind,:), zeros(1,featNumUse-size(ExtSet,2))];
        
    else
        best_cost1(ell,:) = pareto_aux;
        best_cost2(ell,:) = pareto_aux;
        best_cost3(ell,:) = pareto_aux;
        best_cost4(ell,:) = pareto_aux;
        best_coeff1(ell,:) = [ExtSet_pareto, zeros(1,featNumUse-size(ExtSet,2))];
        best_coeff2(ell,:) = [ExtSet_pareto, zeros(1,featNumUse-size(ExtSet,2))];
        best_coeff3(ell,:) = [ExtSet_pareto, zeros(1,featNumUse-size(ExtSet,2))];
        best_coeff4(ell,:) = [ExtSet_pareto, zeros(1,featNumUse-size(ExtSet,2))];
    end
    
end

% For all features (credit dataset)
best_cost1(featNum,:) = EvalFuncs([1:featNum]);
best_cost2(featNum,:) = EvalFuncs([1:featNum]);
best_cost3(featNum,:) = EvalFuncs([1:featNum]);
best_cost4(featNum,:) = EvalFuncs([1:featNum]);

%% Reconstruction errors
for ii = 1:size(best_coeff1,1)
    % PCA
    P = coeff(:,1:ii)*transpose(coeff(:,1:ii)); % Projection matrix for the combination
    approx_A = A*P; % Projection for class A
    approx_B = B*P; % Projection for class B
    approx_M = M*P; % Projection for all classes
    
    recons_A_aux(ii) = re(A, approx_A)/size(A, 1); % Reconstruction error for class A
    recons_B_aux(ii) = re(B, approx_B)/size(B, 1); % Reconstruction error for class B
    recons_error(ii) = re(M, approx_M)/size(M, 1); % Reconstruction error for all classes
    
    fair_meas_recons_cost(ii) = (recons_A_aux(ii)-recons_B_aux(ii))^2; % Calculates the fairness criteria
    
    % Fair PCA
    P = coeff(:,best_coeff1(ii,1:ii))*transpose(coeff(:,best_coeff1(ii,1:ii))); % Projection matrix for the combination
    approx_A = A*P; % Projection for class A
    approx_B = B*P; % Projection for class B
    approx_M = M*P; % Projection for all classes
    
    recons_A_aux_fair1(ii) = re(A, approx_A)/size(A, 1); % Reconstruction error for class A
    recons_B_aux_fair1(ii) = re(B, approx_B)/size(B, 1); % Reconstruction error for class B
    recons_error_fair1(ii) = re(M, approx_M)/size(M, 1); % Reconstruction error for all classes
    
    fair_meas_fair_cost1(ii) = (recons_A_aux_fair1(ii)-recons_B_aux_fair1(ii))^2; % Calculates the fairness criteria
    
    P = coeff(:,best_coeff2(ii,1:ii))*transpose(coeff(:,best_coeff2(ii,1:ii))); % Projection matrix for the combination
    approx_A = A*P; % Projection for class A
    approx_B = B*P; % Projection for class B
    approx_M = M*P; % Projection for all classes
    
    recons_A_aux_fair2(ii) = re(A, approx_A)/size(A, 1); % Reconstruction error for class A
    recons_B_aux_fair2(ii) = re(B, approx_B)/size(B, 1); % Reconstruction error for class B
    recons_error_fair2(ii) = re(M, approx_M)/size(M, 1); % Reconstruction error for all classes
    
    fair_meas_fair_cost2(ii) = (recons_A_aux_fair2(ii)-recons_B_aux_fair2(ii))^2; % Calculates the fairness criteria
    
    P = coeff(:,best_coeff3(ii,1:ii))*transpose(coeff(:,best_coeff3(ii,1:ii))); % Projection matrix for the combination
    approx_A = A*P; % Projection for class A
    approx_B = B*P; % Projection for class B
    approx_M = M*P; % Projection for all classes
    
    recons_A_aux_fair3(ii) = re(A, approx_A)/size(A, 1); % Reconstruction error for class A
    recons_B_aux_fair3(ii) = re(B, approx_B)/size(B, 1); % Reconstruction error for class B
    recons_error_fair3(ii) = re(M, approx_M)/size(M, 1); % Reconstruction error for all classes
    
    fair_meas_fair_cost3(ii) = (recons_A_aux_fair3(ii)-recons_B_aux_fair3(ii))^2; % Calculates the fairness criteria
    
    P = coeff(:,best_coeff4(ii,1:ii))*transpose(coeff(:,best_coeff4(ii,1:ii))); % Projection matrix for the combination
    approx_A = A*P; % Projection for class A
    approx_B = B*P; % Projection for class B
    approx_M = M*P; % Projection for all classes
    
    recons_A_aux_fair4(ii) = re(A, approx_A)/size(A, 1); % Reconstruction error for class A
    recons_B_aux_fair4(ii) = re(B, approx_B)/size(B, 1); % Reconstruction error for class B
    recons_error_fair4(ii) = re(M, approx_M)/size(M, 1); % Reconstruction error for all classes
    
    fair_meas_fair_cost4(ii) = (recons_A_aux_fair4(ii)-recons_B_aux_fair4(ii))^2; % Calculates the fairness criteria
end

%% Figures
checkpoints = 1:length(recons_error);

% Reconstruction error
figure; plot(checkpoints, recons_error,'kx-', checkpoints, recons_error_fair1,'ro-');
legend('PCA','MOFPCA'); xlabel('Number of features'); ylabel('Reconstruction error');

figure; plot(checkpoints, recons_error,'kx-', checkpoints, recons_error_fair2,'ro-');
legend('PCA','MOFPCA'); xlabel('Number of features'); ylabel('Reconstruction error');

figure; plot(checkpoints, recons_error,'kx-', checkpoints, recons_error_fair3,'ro-');
legend('PCA','MOFPCA'); xlabel('Number of features'); ylabel('Reconstruction error');

figure; plot(checkpoints, recons_error,'kx-', checkpoints, recons_error_fair4,'ro-');
legend('PCA','MOFPCA'); xlabel('Number of features'); ylabel('Reconstruction error');

% Fairness measure - PCA and Fair PCA
figure; plot(checkpoints, fair_meas_recons_cost,'kx-', checkpoints, fair_meas_fair_cost1, 'ro--');
legend('PCA','MOFPCA'); xlabel('Number of features'); ylabel('Fairness measure');

figure; plot(checkpoints, fair_meas_recons_cost,'kx-', checkpoints, fair_meas_fair_cost2, 'ro--');
legend('PCA','MOFPCA'); xlabel('Number of features'); ylabel('Fairness measure');

figure; plot(checkpoints, fair_meas_recons_cost,'kx-', checkpoints, fair_meas_fair_cost3, 'ro--');
legend('PCA','MOFPCA'); xlabel('Number of features'); ylabel('Fairness measure');

figure; plot(checkpoints, fair_meas_recons_cost,'kx-', checkpoints, fair_meas_fair_cost4, 'ro--');
legend('PCA','MOFPCA'); xlabel('Number of features'); ylabel('Fairness measure');

% Reconstruction erros for each class (PCA)
figure; plot(checkpoints, recons_A_aux,'g*-', checkpoints, recons_B_aux, 'bs-');
legend('Lower education','Higher education'); xlabel('Number of features'); ylabel('Reconstruction error');

% Reconstruction erros for each class (MOFPCA)
figure; plot(checkpoints, recons_A_aux_fair1,'g*-', checkpoints, recons_B_aux_fair1, 'bs-');
legend('Lower education','Higher education'); xlabel('Number of features'); ylabel('Reconstruction error');

figure; plot(checkpoints, recons_A_aux_fair2,'g*-', checkpoints, recons_B_aux_fair2, 'bs-');
legend('Lower education','Higher education'); xlabel('Number of features'); ylabel('Reconstruction error');

figure; plot(checkpoints, recons_A_aux_fair3,'g*-', checkpoints, recons_B_aux_fair3, 'bs-');
legend('Lower education','Higher education'); xlabel('Number of features'); ylabel('Reconstruction error');

figure; plot(checkpoints, recons_A_aux_fair4,'g*-', checkpoints, recons_B_aux_fair4, 'bs-');
legend('Lower education','Higher education'); xlabel('Number of features'); ylabel('Reconstruction error');

% Pareto front - 10 features
figure; plot(pareto_cost(pareto_numb(10):pareto_numb(10+1)-1,1),pareto_cost(pareto_numb(10):pareto_numb(10+1)-1,2),'kx');
hold on; plot(best_cost1(10,1),best_cost1(10,2),'ro');
legend('Non-dominated','Selected solution'); xlabel('Reconstruction error'); ylabel('Fairness criteria');

% Pareto front - Comparison
figure; plot(pareto_cost(pareto_numb(6):pareto_numb(6+1)-1,1),pareto_cost(pareto_numb(6):pareto_numb(6+1)-1,2),'kx');
hold on; plot(recons_error(6),fair_meas_recons_cost(6),'ro');
hold on; plot(best_cost4(6,1),best_cost4(6,2),'go');
hold on; plot(best_cost2(6,1),best_cost2(6,2),'bo');
legend('Non-dominated solutions','PCA solution','The fairest solution','Selected solution'); xlabel('Reconstruction error'); ylabel('Fairness criteria');
