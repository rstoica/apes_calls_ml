function [ net ] = feedForward ( net, input )
% updateUnites Updates the values of each units in the input, hidden layer
% and output based on new incoming inputs and the architecture of a
% classical feedforward neural network
% 
%   Inputs : *net = neural network to be updated
%            *input = input vector presented to the network (row vector)
%   Outputs: *net = updated neural network structure - output can be
%            recovered as net.y

    % add bias unit to the input and then store it in the first layer
    net.x = input';
    net.xb = [1;net.x];

    % compute hidden activations from first layer, weights and act. function
    net.h = net.Wh * net.xb;
    % add bias hidden unit
    net.hb = [1;net.func(net.h)];

    % compute hidden activations from hidden layer, weights and act. function
    net.y = net.func(net.Wo * net.hb);
end