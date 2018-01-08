%% Data Aquicition
clear all;
clc;
clf;

test_no = 100;  % number of testing data
eg = 0.09;  % sum-squared error goal
sc = 1;   % spread constant

inp = xlsread('cleveland_database_14_v6.xlsx');
[nrows, ncols] = size(inp);

tar = inp(:,ncols);

weight = 1./var(inp(:,1:ncols-1));

[coeff,scores,latent,~,exp] = pca(inp(:,1:ncols-1),'VariableWeights',weight);

inp = [scores(:,1:3),tar];
[nrows, ncols] = size(inp);


rng('shuffle');
inp = inp(randperm(nrows),:);


train_no = nrows - test_no;
label = inp(1:train_no,ncols);
data = inp(1:train_no,1:ncols-1);
test = inp(train_no+1:nrows,1:ncols-1);


%% Radial Basis Funciton
net_radial = newrb(data',label',eg,sc);
result_radial = [round(net_radial(test'));inp(nrows-test_no+1:nrows,ncols)']';
%result1 = [floor(net(test'));inp(nrows-test_no+1:nrows,ncols)']';

sum(result_radial(:,1)==result_radial(:,2))
%sum(result1(:,1)==result(:,2))

%%