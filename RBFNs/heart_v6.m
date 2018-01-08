% RBF Neural Networks (Parameters are selected using K-means Clustering)
% Parameter (K: Number of Kernels)

% RBFNN have 5 parameters for optimization: 
% 1- The weights between the hidden layer and the output layer. 
% 2- The activation function. 
% 3- The center of activation functions. 
% 4- The distribution of center of activation functions. 
% 5- The number of hidden neurons. 

% The weights between the hidden layer and the output layer are calculated by using Moore-Penrose generalized pseudo-inverse. This algorithm overcomes many issues in traditional gradient algorithms such as stopping criterion, learning rate, number of epochs and local minima. Due to its shorter training time and generalization ability, it is suitable for real-time applications. 
% The radial basis function selected is usually a Gaussian kernel for pattern recognition application. 
% Generally the center and distribution of activation functions should have characteristic similar to data. Here, the center and width of Gaussians are selected using Kmeans clustering algorithm. 
% Based on universal approximation theory center and distribution of activation functions are not deterministic if the numbers of hidden neurons being sufficient enough, one can say that the single hidden layer feed-forward network with sufficient number of hidden neurons can approximate any function to any arbitrary level of accuracy.

clc
clear all
close all


heart_data = [xlsread('cleveland_database_14_v5.xlsx',1,'A2:J570')];
heart_class = [xlsread('cleveland_database_14_v5.xlsx',1,'K2:K570')];

heart_data_together = [heart_data, heart_class];
heart_data_together = heart_data_together(randperm(569),:);

%Create backpropagation based neural network
heart_data_train = heart_data_together(1:350,1:10);
heart_data_test = heart_data_together(351:569,1:10);
heart_class_test_output = heart_data_together(351:569,11);
heart_class_train = heart_data_together(1:350,11);

Fr = heart_data_train;
Fs = heart_data_test;
Lr = heart_class_train;
Ls = heart_class_test_output;%% Generate Points & Labels

K               = 80;                               % Number of Clusters (Number of Kernels)
KMI             = 100;                              % K-means Iteration

[W, MU, SIGMA]  = rbfn_train(Fr, Lr, K, KMI);      % train RBFNs
Y               = rbfn_test(Fs, W, K, MU, SIGMA);  % test RBFNs

hold on
plot(Fs(Ls == 0, 1), Fs(Ls == 0, 2), '.')
plot(Fs(Ls == 1, 1), Fs(Ls == 1, 2), '.r')
plot(Fs(Y  == 0, 1),  Fs(Y == 0, 2), 'o')
plot(Fs(Y  == 1, 1),  Fs(Y == 1, 2), 'or')
hold off
legend('original data: class 1','original data: class 2','RBFNS test: class 1','RBFNS test: class 2')
grid on

SR   = 1 - sum(abs(Y-Ls))/size(Y,1);
disp(strcat('Classification accuracy =', num2str(SR * 100), '%'))
