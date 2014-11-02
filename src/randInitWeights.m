function [ W ] = randInitWeights(rows, cols)
% initWeights Initiates the weights of a network randmly based on a Gaussian
% distribution of mean mu and standard deviation sigma.
%   Inputs : *rows = number of rows of the weights matrix
%            *cols = number of cols of the weights matrix
%   Outputs: *W = randomly initialized weights matrix with entries between
%            -0.25 and 0.25;
    epsilon = 0.25;
    W = rand(rows,cols) * (2 * epsilon) - epsilon; 
end