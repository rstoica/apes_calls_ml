function [ L grad ] = costFunction( weights,net,X,Y,lambda )
% costFunction Implements the cost function for the neural network.
%
% Computes its values given the arguments the weights of network and also
% calculates the gradient of the cost function, such that this can be
% passed as a handle to the fmincg.m function for solving the optimization
% problems with respect to the weights of the network.
%   INPUTS: *weights = weights of the network as a long column vector to be
%   unrolled into weight matrices
%           *net = neural network structure
%           *X = input data sample points
%           *Y = output target data as vectors of 0s and 1s
%           *lambda = regularization term for handling overfitting
%   OUTPUTS: *L = value of the cost function over all the sample points
%            given the weights
%            *grad = column rolled up gradient of the cost function given
%            weights of neural network

    % unroll weights into matrices
    net.Wh = reshape(weights(1:net.Nh*(net.Ni+1)),net.Nh,net.Ni+1);
    net.Wo = reshape(weights((net.Nh*(net.Ni+1))+1:end),net.No,net.Nh+1);
    M = size(X,1);
    L = 0;
    % set gradient accumulators to 0
    net.DWh = zeros(size(net.Wh));
    net.DWo = zeros(size(net.Wo));
    % commpute cost function values 
    for i=1:M
        net = feedForward(net,X(i,:));
        L = L + sum((Y(i,:)*log(net.y))+(1-Y(i,:))*log(1-net.y));
    end
    L = -L / M;
    % regularize cost
    regularizer = lambda/2/M*(sum(sum(net.Wh(:,2:end).^2)) + sum(sum(net.Wo(:,2:end).^2)));
    L = L + regularizer;
    % backpropagation and computation of gradient for the weights
    for i=1:M
        net = feedForward(net,X(i,:));
        delta_o = net.y - Y(i,:)';
        delta_h = (net.Wo'*delta_o).*net.dfunc([1;net.h]);
        net.DWh = net.DWh + delta_h(2:end) * net.xb';
        net.DWo = net.DWo + delta_o * net.hb'; 
    end
    % handle regularization term of the cost function at the weights level
    % do not add bias terms -> start from second column only
    net.DWh = 1/M*(net.DWh + lambda * [zeros(size(net.DWh,1),1),net.DWh(:,2:end)]);
    net.DWo = 1/M*(net.DWo + lambda * [zeros(size(net.DWo,1),1),net.DWo(:,2:end)]);
    % roll gradient into column format
    grad = [net.DWh(:);net.DWo(:)];
end