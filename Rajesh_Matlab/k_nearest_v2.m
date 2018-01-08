% KNN using fitcknn

%k = 10;  % no. of nearest neighbours
test_no = 20;  % number of testing data

inp = xlsread('cleveland_database_14_v3.xlsx');
[nrows, ncols] = size(inp);
inp = inp(randperm(nrows),:);

train_no = nrows - test_no;
label = inp(1:train_no,ncols);
data = inp(1:train_no,1:ncols-1);
test = inp(train_no+1:nrows,1:ncols-1);
%idx = knnsearch(data,test,'K',k);
%near = zeros(size(idx));

rng(1);
knn = fitcknn(data,label,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus'));
result = [predict(knn,test),inp(nrows-test_no+1:nrows,ncols)];

CrossValModel = crossval(knn);
classError = kfoldLoss(CrossValModel)

sum(result(:,1)==result(:,2))