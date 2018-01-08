%% Data Acquisition
clear all;
clc;
clf;
close all;

test_no = 100;  % number of testing data

inp = xlsread('cleveland_database_14_v6.xlsx');
[nrows, ncols] = size(inp);

tar = inp(:,ncols);

%weight = 1./var(inp(:,1:ncols-1));

weight = 1./xcorr(inp(:,1:ncols-1));

[coeff,scores,latent,~,exp] = pca(inp(:,1:ncols-1),'VariableWeights','variance');

new_feature = scores';
group1 = new_feature(:,1:323);
group22 = new_feature(:,324:480);

%PCA plotting
figure
plot(group1(1,:), group1(2,:), 'b.', group22(1,:), group22(2,:), 'r.');
figure
plot3(group1(1,:),group1(2,:),group1(3,:),'b.',group22(1,:),group22(2,:),group22(3,:),'r.');

inp = [scores(:,1:3),tar];
[nrows, ncols] = size(inp);


rng('shuffle');
inp = inp(randperm(nrows),:);


train_no = nrows - test_no;
label = inp(1:train_no,ncols);
data = inp(1:train_no,1:ncols-1);
test = inp(train_no+1:nrows,1:ncols-1);

Acc = zeros(1,4);


%% Radial Basis Funciton
eg = 0.09;  % sum-squared error goal
sc = 1;   % spread constant

net_radial = newrb(data',label',eg,sc);
result_MLFF = [round(net_radial(test'));inp(nrows-test_no+1:nrows,ncols)']';
%result1 = [floor(net(test'));inp(nrows-test_no+1:nrows,ncols)']';

Acc(1) = sum(result_MLFF(:,1)==result_MLFF(:,2))/test_no*100;
RBF_accuracy = Acc(1)
%sum(result1(:,1)==result(:,2))

%% Multilayered Feed Forward Network

net_back = feedforwardnet(60,'trainlm'); %rule of thumb -- sqrt(75) ## of input vectors

while(1)    
    [net_back, tr] = train(net_back,data',label');
    output = net_back(data');
    %check the performance as condition for exiting
    perf = perform(net_back,label',output)
    if(perf <= 0.13)
        break;
    end
end

result_MLFF = [round(net_back(test'));inp(nrows-test_no+1:nrows,ncols)']';
%result1 = [floor(net(test'));inp(nrows-test_no+1:nrows,ncols)']';

Acc(2) = sum(result_MLFF(:,1)==result_MLFF(:,2))/test_no*100;
MLFN_accuracy = Acc(2)
%sum(result1(:,1)==result(:,2))


%% K-nearest neibour
knn = fitcknn(data,label,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus'));
result_knn = [predict(knn,test),inp(nrows-test_no+1:nrows,ncols)];

CrossValModel = crossval(knn);
classError = kfoldLoss(CrossValModel)

Acc(3) = sum(result_knn(:,1)==result_knn(:,2))/test_no*100;
KNn_accuracy = Acc(3)

%% Deceision Tree

tree = fitctree(data,label,'OptimizeHyperparameters','auto');
result = [predict(tree,test),inp(nrows-test_no+1:nrows,ncols)];
view(tree,'Mode','graph')

%rng(1);
classErrorByCrossValidation = cvloss(tree)

Acc(4) = sum(result(:,1)==result(:,2))/test_no*100;
DT_accuracy = Acc(4)

%crossValuation = crossval(tree,label)
%classError7 = kfoldLoss(tree)
