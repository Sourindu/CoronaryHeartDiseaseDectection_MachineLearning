clear all;
clc
clf
close all;

heart_data = [xlsread('cleveland_database_14_v5b.xlsx',1,'A2:J570')];
heart_class = [xlsread('cleveland_database_14_v5b.xlsx',1,'K2:K570')];
%[~,heart_class_name] = xlsread('cleveland_database_14_v5.xlsx',1,'O2:O298');

% heart2 = [];
% 
% for x = 1:10
%     fac = sqrt(sum(heart_data(:,x).^2));
%     heart2(:,x) = heart_data(:,x)./fac;
% end

%PCA implementation for dimension reductionality

weight = 1./var(heart_data);
[coeff,scores,latent,~,exp] = pca(heart_data,'VariableWeights',weight);
scores = scores';


group1 = scores(:,1:323);
group22 = scores(:,324:569);
%  group2 = scores(:,324:380);
%  group3 = scores(:,381:516);
%  group4 = scores(:,517:556);
%  group5 = scores(:,557:569);

%PCA plotting

% gscatter3(x,y,z);
figure
plot(group1(1,:), group1(2,:), 'b.');
hold on 
plot(group22(1,:), group22(2,:), 'r.');
hold off;

% 
% figure;
% plot3(group1(1,:),group1(2,:),group1(3,:),'b.',group2(1,:),...
%     group2(2,:),group2(3,:),'r.',group3(1,:),group3(2,:),group3(3,:),'gx',...
%     group4(1,:),group4(2,:),group4(3,:),'m+',group5(1,:),group5(2,:),group5(3,:),'k.');

figure;
plot3(group1(1,:),group1(2,:),group1(3,:),'b.',group22(1,:),group22(2,:),group22(3,:),'r.');

% %rng(1);

% 
% % 
% % %KNN Classification
% % rng(1)
% % BestHeartModel = fitcknn(heart_data,heart_class_name,'OptimizeHyperparameters','auto',...
% %     'HyperparameterOptimizationOptions',...
% %     struct('AcquisitionFunctionName','expected-improvement-plus'))
% % 
% % rng(1)
% % BestHeartModel_reduced = fitcknn(heart_data_reduced,heart_class_name,'OptimizeHyperparameters','auto',...
% %     'HyperparameterOptimizationOptions',...
% %     struct('AcquisitionFunctionName','expected-improvement-plus'))
% % 
% % 
% % rng(1); % For reproducibility
% % CrossValModel = crossval(BestHeartModel);
% % classError = kfoldLoss(CrossValModel)
% % 
% % rng(1); % For reproducibility
% % CrossValModel = crossval(BestHeartModel_reduced);
% % classError_reduced = kfoldLoss(CrossValModel)
% 
% %rng(1)
% 
% heart_data_together = [heart_data, heart_class];
% heart_data_together = heart_data_together(randperm(569),:);
% 
% 
% %Create backpropagation based neural network
% net = feedforwardnet(35,'trainlm'); %rule of thumb -- sqrt(75) ## of input vectors
% heart_data_train = heart_data_together(1:350,1:10);
% heart_data_test = heart_data_together(351:569,1:10);
% heart_class_test_output = heart_data_together(351:569,11);
% heart_class_train = heart_data_together(1:350,11);
% 
% %train the neural network
% while(1)    
%     [net, tr] = train(net,heart_data_train',heart_class_train');
%     output = net(heart_data_train');
%     %check the performance as condition for exiting
%     perf = perform(net,heart_class_train',output)
%     if(perf <= 0.22)
%         break;
%     end
% end
% 
%Backpropagation With Reduced Feature

heart_data_reduced = [scores(1:3,:)',heart_class];
heart_data_reduced = heart_data_reduced(randperm(569),:)

%Create backpropagation based neural network
net = feedforwardnet(40,'trainlm'); %rule of thumb -- sqrt(75) ## of input vectors
heart_data_train = heart_data_reduced(1:350,1:3);
heart_data_test = heart_data_reduced(351:569,1:3);
heart_class_test_output = heart_data_reduced(351:569,4);
heart_class_train = heart_data_reduced(1:350,4);

%train the neural network
while(1)    
    [net, tr] = train(net,heart_data_train',heart_class_train');
    output = net(heart_data_train');
    %check the performance as condition for exiting
    perf = perform(net,heart_class_train',output)
    if(perf <= 0.1)
        break;
    end
end


% 
% %Testing the neural network
% outputfinal = net(heart_data_test');
% in1 = round(vec2ind(heart_class_test_output'));
% in2 = round(vec2ind(outputfinal));
% %Percentage of Error
% perErr = sum(in1 ~= in2)/numel(in1)%Calulation of the percentage Error
% %Plotting the confucion matrix to check accuracy of the result
% figure(1), plotconfusion(data_target_test,outputfinal)
% 
