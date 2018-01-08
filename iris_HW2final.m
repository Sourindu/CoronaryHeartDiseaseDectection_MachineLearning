%[x,t] = iris;
% net = feedforwardnet(10,'trainlm');
% net = train(net,iris);
%%y = net(x)
%fac = [sqrt(sum(iris(:,1).^2)),sqrt(sum(iris(:,2).^2)),sqrt(sum(iris(:,3).^2)),sqrt(sum(iris(:,4).^2))]
clear
clc
iris = [xlsread('iris.xlsx',1,'A1:D150'),xlsread('iris.xlsx',1,'F1:F150')];
%normalization of data
fac = [sqrt(sum(iris(:,1).^2)),sqrt(sum(iris(:,2).^2)),sqrt(sum(iris(:,3).^2)),sqrt(sum(iris(:,4).^2))];
for c = 1:4
    iris(:,c) = iris(:,c)./fac(c);
end
%normalization ends
%Creating training data and target
data_train = [iris(1:25,1:4);iris(51:75,1:4);iris(101:125,1:4)]';
data_target = [iris(1:25,5);iris(51:75,5);iris(101:125,5)]';
%Creating testing data and target
data_target_test = [iris(76:100,5);iris(26:50,5);iris(126:150,5)]';
data_test = [iris(76:100,1:4);iris(26:50,1:4);iris(126:150,1:4)]';
%create backpropagation based neural network
net = feedforwardnet(8,'trainlm'); %rule of thumb -- sqrt(75) ## of input vectors

while(1)
    %train the neural network
    [net,tr] = train(net,data_train,data_target);
    output = net(data_train);
    %check the performance as condition for exiting
    perf = perform(net,data_target,output)
    if(perf <= 0.01)
        break;
    end
            
end
%Testing the neural network
outputfinal = net(data_test);
in1 = round(vec2ind(data_target_test'));
in2 = round(vec2ind(outputfinal'));
%Percentage of Error
perErr = sum(in1 ~= in2)/numel(in1)%Calulation of the percentage Error
%Plotting the confucion matrix to check accuracy of the result
figure(1), plotconfusion(data_target_test,outputfinal)
%Plotting validation for trained neural network
figure(2), plottrainstate(tr)
figure(3), plotperform(tr)