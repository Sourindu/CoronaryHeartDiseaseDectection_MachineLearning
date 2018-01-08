% Project
% K-Nearest Neighbour Algorithm
clear 
clc
k = 10;  % no. of nearest neighbours
test_no = 10;  % number of testing data

%inp = xlsread('cleveland_database_14.xlsx');
inp = [xlsread('cleveland_database_14.xlsx',1,'A2:M298')];
[nrows, ncols] = size(inp);
inp = inp(randperm(nrows),:);
% data = zeros(nrows-test_no,ncols);  % normalized values
% for i = 1:ncols
%     data(:,i) = ((inp(:,i)-min(inp(:,i)))/(max(inp(:,i))-min(inp(:,i))));
% end

train_no = nrows - test_no;
result = inp(1:train_no,ncols);
data = inp(1:train_no,1:ncols-1);
test = inp(train_no+1:nrows,1:ncols-1);
dist = zeros(train_no,test_no);

for i = 1:test_no
    for j = 1:train_no
        dist(j,i) = sqrt(sum([data(j,:)-test(i,:)].^2));
    end
end

[dist_sort,idx] = sort(dist);
% for i = 1:test_no
%     new_result = result(idx(:,i));
% end

new_result1 = result(idx(:,1));
new_result2 = result(idx(:,2));
