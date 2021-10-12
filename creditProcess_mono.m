function [M, A, B, A_orig, B_orig] = creditProcess_mono()

% Preprocess the Taiwanese Credit dataset and return the centered/normalized data matrix M (with al samples)
% and the samples of each sensitive group A and B. 

addpath data/credit

data = csvread('default_degree.csv', 2, 1);

% Vector of sensitive attributes
sensitive = data(:,1);

% Define the sensitive attribute: 0 for grad school and university and >0 for high school, other
normalized = (sensitive-1).*(sensitive-2);

% Extract the sensitive attribute from the dataset
data = data(:,2:end);

[m,n] = size(data); % Number of attributes

% Centering and normalizing the dataset
data = (data - repmat(mean(data),m,1))./repmat(std(data),m,1);

% Data for low educated populattion
data_lowEd = data(find(normalized),:);

% date for high educated population
data_highEd = data(find(~normalized),:);

% Centering and normalizing the sensitive attributes
data_lowEd = (data_lowEd - repmat(mean(data_lowEd),size(data_lowEd,1),1));
data_highEd = (data_highEd - repmat(mean(data_highEd),size(data_highEd,1),1));

M = data;
A_orig = M(find(normalized),:);
B_orig = M(find(~normalized),:);
A = data_lowEd;
B = data_highEd;
    
end
