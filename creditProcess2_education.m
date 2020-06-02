function [M, A, B] = creditProcess2_education()

% preprocess the credit data. The output of the function is the centered
% data as matrix M. Centered low educated group A and high educated as
% group B. 

addpath data/credit

data = csvread('default_degree.csv', 2, 1);

% vector of sensitive attribute.
sensitive = data(:,1);

% normalizing the sensitive attribute vetor to have 0 for grad school and 
% university level education and positive value for high school, other
normalized = (sensitive-1).*(sensitive-2);

% getting rid of the colum corresponding to the senstive attribute.
data = data(:,2:end);

n = size(data, 2);

% centering the data and normalizing the variance across each column
for i=1:n
   data(:,i) = data(:,i) - mean(data(:,i));
   data(:,i) = data(:,i)/std(data(:,i));
end

% data for low educated populattion
data_lowEd = data(find(normalized),:);

% date for high educated population
data_highEd = data(find(~normalized),:);

M = data;
A = data_lowEd;
B = data_highEd;
    

end
