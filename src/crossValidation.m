% Loads and works on the training data set in order to train a feedforward
% neural network classifier and run cross-validation on it

% clear previous workspace contents
clc;
clear all;
apeCallsDataImport

% save training data to file for later reuse
save('datatrain.mat','trainInputs','trainOutputs');
clear all;

load datatrain.mat
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

% perform cross-validation (takes a lot of time ~ 10h)
% 7-fold cross-validation
K = 7;
% vary number of units in the hidden layer
H = (10:2:30)';
HH = length(H);
% vary regularization coefficient
lambda = [0,logspace(-1,1,19)]';
LL = length(lambda);
% store training errors for H and lambda value pairs
trainErr = zeros(HH,LL);
% store validation errors for H and lambda value pairs
valErr = zeros(HH,LL);

% run cross-validation
for h=1:HH
    for l=1:LL
        total_train_err = 0;
        total_val_err = 0;
        for k=1:K
            fprintf('H = %d, l = %.2f, K = %d\n',H(h),lambda(l),k);
            Xtrain = X;
            Ytrain = Y;
            % select utter. every 7 frames with offset from 1 to 7
            Xval = Xtrain(k:7:size(X,1),:);
            Yval = Ytrain(k:7:size(Y,1),:);
            % remove selected utter. from training set
            Xtrain(k:7:size(X,1),:) = [];
            Ytrain(k:7:size(X,1),:) = [];
            % init and train nn
            net = initFFNN(size(Xtrain,2),H(h),size(Ytrain,2));
            net = trainFFNN(net,Xtrain,Ytrain,lambda(l));
            % validate on test data
            val_err = 0;
            for i=1:size(Xval,1)
                class = classifyApe(net,Xval(i,:));
                target = find(Yval(i,:)==1);
                if (class~=target)
                    val_err = val_err + 1;
                end
            end
            total_val_err = total_val_err + val_err/size(Xval,1)*100;
            % test on training data
            train_err = 0;
            for i=1:size(Xtrain,1)
                class = classifyApe(net,Xtrain(i,:));
                target = find(Ytrain(i,:)==1);
                if (class~=target)
                    train_err = train_err + 1;
                end
            end
            total_train_err = total_train_err + train_err/size(Xtrain,1)*100;
        end
        % statistics of misclassification errors
        total_val_err = total_val_err / K;
        total_train_err = total_train_err / K;
        trainErr(h,l) = total_train_err;
        valErr(h,l) = total_val_err;
    end
end

% get min error and values where is recorded
minErr = min(min(valErr));
[Hopt,lambdaopt] = find(valErr==minErr);
Hopt = H(Hopt)  % Hopt = 24;
lambdaopt = lambda(lambdaopt)   % lambdaopt = 0