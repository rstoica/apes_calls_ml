function [ net ] = initFFNN(input_dim, hidden_dim, output_dim)
%initFFNN Creates a feedforward neural network with one hidden layer
%   
%   Inputs: *input_dim = number of input units for the first layer (without
%           bias input unit)
%           *hidden_dim = number of units available in the hidden layer
%           (without bias unit)
%           *output_dim = number of output units for the output layer
%           *func = function handle for the unit "excitation" activity
%
%   Outputs: *net = neural network as a struct data type with the following
%   fields:
%               .Ni = number of input units without bias unit
%               .No = number of output units
%               .Nh = number of hidden units in the second layer without
%               bias unit
%               .x = input layer unit values without bias component
%               .xb = input layer unit values as column vector, together
%               with bias unit positioned on the first position
%               .h = activation values of the hidden units without bias
%               .hb = activation values of the hidden units, together with
%               bias unit positioned first
%               .y = output unit values as a column vector
%               .func = function handle for determining unit activity
%               as logistic sigmoid function
%               .dfunc = function handle for the the derivative of the
%               network activation function
%               .Wh = weights from input to hidden layer
%               .DWh = partial derivatives of cost function wrt 
%               weights from input to hidden layer 
%               .Wo = weights from hidden to output layer
%               .DWo = partial derivatives of cost function wrt weights
%               from hidden to output layer

    % store input dimension
    net.Ni = input_dim;

    % store output dimension
    net.No = output_dim;

    % store number of hidden units
    net.Nh = hidden_dim;

    % units book-keeping
    net.x = zeros(net.Ni, 1);
    net.xb = zeros(net.Ni + 1, 1);
    net.h = zeros(net.Nh, 1);
    net.hb = zeros(net.Nh + 1, 1);
    net.y = zeros(net.No , 1);

    % randomly initiate weights given a Gaussian distribution of mean 0 and
    % standard deviation 0.25;
    net.Wh = zeros(net.Nh, net.Ni + 1);
    net.Wo = zeros(net.No, net.Nh + 1);
    net.DWh = zeros(net.Nh, net.Ni + 1);
    net.DWo = zeros(net.No, net.Nh + 1);

    % activation function is the logistic sigmoid
    net.func = @(z)(1.0 ./ (1.0 + exp(-z)));
    net.dfunc = @(z)(exp(-z)./(1.0 + exp(-z)).^2);
end