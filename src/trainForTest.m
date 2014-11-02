function [ net ] = trainForTest( Hopt, lambdaopt )
%trainForTest Utility function to quickly create a neural network
%classifier using the optimal H and optimal lambda parameters
%   
%   Depends on: 
%       apeCallsDataImport
%       initFFNN
%       trainFFNN
%       feedForward
%       randInitWeights
%       costFunction
%       fmincg

    apeCallsDataImport
    N = length(trainInputs);

    % compute minimum recording length over the entire training data set
    min_rec = size(trainInputs{1},1);
    idx = 1;
    for i = 2:N
        rec_len = size(trainInputs{i},1);
        if (min_rec > rec_len)
            min_rec = rec_len;
            idx = i;
        end
    end
    % printout result
    fprintf('The minimal recording length is of %d discrete frames recorded during rec. #%d\n',min_rec,idx);
    % result is min_rec = 7 at recording 69

    % truncate training sample points to min recording length
    X = [];
    Y = [];
    for i = 1:N
        trainInputs{i} = trainInputs{i}(1:min_rec,:);
        X = [X ; trainInputs{i}];
        trainOutputs{i} = trainOutputs{i}(1:min_rec,:);
        Y = [Y ; trainOutputs{i}];
    end
    net = initFFNN(size(X,2),Hopt,size(Y,2));
    net = trainFFNN(net,X,Y,lambdaopt);
end

