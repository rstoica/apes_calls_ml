function [ class ] = classifyApe( net,input )
%classifyApe Given one input sample recording it classifies the ape that
%produced it in one of the 9 considered classes
%
%   Implements a majority rule of the classification at the frame level of
%   the input recording. Thus, the class with the most frames classified to
%   it wins.
%   INPUTS: *net = neural network struct
%           *input = input data sample point
%   OUTPUTS:*class = class to which the input is classified to

    % no of frames in the input data sample point
    l = size(input,1);
    % accumulator for class decision
    accum = zeros(9,1);
    % check all frames and decide to which class they belong to
    for i = 1:l
        % feed frame forward
        % add bias unit to the input and then store it in the first layer
        net.x = input(i,:)';
        net.xb = [1;net.x];
        % compute hidden activations from first layer, weights and act. function
        net.h = net.Wh * net.xb;
        % add bias hidden unit
        net.hb = [1;net.func(net.h)];
        % compute hidden activations from hidden layer, weights and act. function
        net.y = net.func(net.Wo * net.hb);
        [~, class_frame] = max(net.y);
        accum(class_frame) = accum(class_frame) + 1;
    end
    % get majority class
    [~, class] = max(accum);
end