function [ net ] = trainFFNN( net, inputs, targets, lambda )
% trainFFNN Trains a neural network using the backpropagation algorithm
% with the conjugate gradient optimization solver
%
%   It initializes the weights and calls the fmincg.m function on the cost
%   function of the network starting from the initial weights to find the
%   optimum ones
%   INPUTS: *net = neural network structure
%           *inputs = input data sample points
%           *outputs = expected target data points (as 1s and 0s filled
%           vectors
%   OUTPUTS:*net = modified neural network structure with updated optimal
%   weights

    % set solver options
    options = optimset('LargeScale','on','MaxIter',200);
    % implement hook to cost function with onl one parameter for fmincg to
    % call
    costFunctionShort = @(W) costFunction(W,net,inputs,targets,lambda);
    % randomly initialize weights of the nn
    Wh = randInitWeights(net.Nh,net.Ni+1);
    Wo = randInitWeights(net.No,net.Nh+1);
    % save them to nn struct
    net.Wh = Wh;
    net.Wo = Wo;
    % roll them in big column vector
    initial_weights = [net.Wh(:);net.Wo(:)];
    % run fmincg optimizer function to find optimal weights
    [weights_opt, cost] = fmincg(costFunctionShort,initial_weights,options);
    % get optimal weights and store them intro nn struct
    net.Wh = reshape(weights_opt(1:net.Nh*(net.Ni+1)),net.Nh,net.Ni+1);
    net.Wo = reshape(weights_opt((net.Nh*(net.Ni+1)+1):end),net.No,net.Nh+1);
end
