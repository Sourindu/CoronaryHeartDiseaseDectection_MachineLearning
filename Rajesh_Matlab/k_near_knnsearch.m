% k nearest neighbor using knnsearch


k = 10;  % no. of nearest neighbours
test_no = 20;  % number of testing data

inp = xlsread('cleveland_database_14_v3.xlsx');
[nrows, ncols] = size(inp);
inp = inp(randperm(nrows),:);

train_no = nrows - test_no;
label = inp(1:train_no,ncols);
data = inp(1:train_no,1:ncols-1);
test = inp(train_no+1:nrows,1:ncols-1);
idx = knnsearch(data,test,'K',k);
near = zeros(size(idx));
result = zeros(test_no,1);

for i = 1:test_no
    near(i,:) = label(idx(i,:)');
    result(i) = mode(near(i,:),2);
end

result(:,2) = inp(nrows-test_no+1:nrows,ncols);

% disp('The result is\n');
% disp(result);
disp('The correct values are:\n');
disp(sum(result(:,1)==result(:,2)));