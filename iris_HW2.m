%[x,t] = iris;
% net = feedforwardnet(10,'trainlm');
% net = train(net,iris);
%%y = net(x)
%fac = [sqrt(sum(iris(:,1).^2)),sqrt(sum(iris(:,2).^2)),sqrt(sum(iris(:,3).^2)),sqrt(sum(iris(:,4).^2))]
clear
clc
iris = [xlsread('iris.xlsx',1,'A1:D150'),xlsread('iris.xlsx',1,'F1:F150')];
%normalization of data
fac = [sqrt(sum(iris(:,1).^2)),sqrt(sum(iris(:,2).^2)),sqrt(sum(iris(:,3).^2)),sqrt(sum(iris(:,4).^2))]
for c = 1:4
    iris(:,c) = iris(:,c)./fac(c);
end
%normalization ends
data_train = [iris(1:25,1:4);iris(51:75,1:4);iris(101:125,1:4)]';
data_target = [iris(1:25,5);iris(51:75,5);iris(101:125,5)]';

data_target_output = [iris(76:100,5);iris(26:50,5);iris(126:150,5)]';
data_test = [iris(76:100,1:4);iris(26:50,1:4);iris(126:150,1:4)]';

net = feedforwardnet(8,'trainlm'); %rule of thumb -- sqrt(75) ## of input vectors

while(1)
    [net,tr] = train(net,data_train,data_target);
    output = net(data_train);

    perf = perform(net,data_target,output)
    if(perf <= 0.01)
        break;
    end
            
end

outputfinal = net(data_test)
plotconfusion(data_target_output,outputfinal)