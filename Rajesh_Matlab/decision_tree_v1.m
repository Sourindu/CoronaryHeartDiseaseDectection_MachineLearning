% decision tree

test_no = 10;  % number of testing data

inp = xlsread('cleveland_database_14_v2.2.xlsx');
%inp = heart_data_reduced;
[nrows, ncols] = size(inp);
inp = inp(randperm(nrows),:);

train_no = nrows - test_no;
label = inp(1:train_no,ncols);
data = inp(1:train_no,1:ncols-1);
test = inp(train_no+1:nrows,1:ncols-1);

tree = fitctree(data,label,'OptimizeHyperparameters','auto');
result = [predict(tree,test),inp(nrows-test_no+1:nrows,ncols)];
%view(tree,'Mode','graph')

classErrorByCrossValidation = cvloss(tree)

sum(result(:,1)==result(:,2))
%classError7 = kfoldLoss(tree)