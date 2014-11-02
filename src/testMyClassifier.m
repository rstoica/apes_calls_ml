% Loads and works on the testing data only in order to evaluate the quality
% of the classifier through its misclassification percentage

% clear all previous workspace variables and contents
clc;
clear all;

% call data import and get test data
apeCallsDataImport

% save test data for later reuse
save('datatest.mat','testInputs','testOutputs');
% remove everything else from workspace
clear all;
% load datatest
load datatest.mat
N = length(testInputs);

% load neural network classifier and use it
load opt_net.mat
min_rec = 7;    % minimum recording length of the training set = 7 at 
% sample recording 69

% if you want to train a new neural network uncomment the 3 lines below
% Hopt = 24;
% lambdaopt = 0;
% nn = trainForTest(Hopt,lambdaopt); 

% testing on the testing data set
Xtest = [];
Ytest = [];
% trim testing data set to the minimal recording length recorded for the
% training. if shorter than that, just add frames of 0s
for i=1:N
    if (size(testInputs{i},1) < min_rec)
        testInputs{i} = [testInputs{i};zeros(min_rec-size(testInputs{i},1),12)];
    else
        testInputs{i} = testInputs{i}(1:min_rec,:);
    end
    Xtest = [Xtest; testInputs{i}];
    if (size(testOutputs{i},1) < min_rec)
        testOutputs{i} = [testOutputs{i};repmat(testOutputs{i}(end,:),min_rec-size(testOutputs{i},1),1)];
    else
        testOutputs{i} = testOutputs{i}(1:min_rec,:);
    end
    Ytest = [Ytest; testOutputs{i}];
end
% compute error on the entire training set and display it in percentage
err = 0;
for i=1:N
    input = Xtest((i-1)*min_rec+1:i*min_rec,:);
    output = find(Ytest((i-1)*min_rec+1,:)==1);
    classified = classifyApe(nn,input);
    if (classified~=output)
        err = err + 1;
    end
end
fprintf('Misclassification percentage on testing dataset is %.3f %%\n',err/N*100);