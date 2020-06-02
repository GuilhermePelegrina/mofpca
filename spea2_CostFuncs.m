
function z=spea2_CostFuncs(theta,coeff,A,B)

P = coeff(:,theta)*transpose(coeff(:,theta)); % Projection matrix for the combination
M = [A;B];

approx_A = A*P; % Projection for class A
approx_B = B*P; % Projection for class B
approx_M = [approx_A; approx_B]; % Projection for all classes

recons_A_aux = re(A, approx_A)/size(A, 1); % Reconstruction error for class A
recons_B_aux = re(B, approx_B)/size(B, 1); % Reconstruction error for class B
recons_M_aux = re(M, approx_M)/size(M, 1); % Reconstruction error for all classes

fair_min_aux = (recons_A_aux-recons_B_aux).^2; % Calculates the fairness criteria

z = [recons_M_aux,fair_min_aux];
end