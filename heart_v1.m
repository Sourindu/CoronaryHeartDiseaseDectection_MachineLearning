clear
clc
iris = [xlsread('iris.xlsx',1,'A1:D150'),xlsread('iris.xlsx',1,'F1:F150')];
heart = [xlsread('cleveland_database_14.xlsx',1,'A2:M298')];
heart2 = [];
for x = 1:13
    fac = sqrt(sum(heart(:,x).^2));
    heart2(:,x) = heart(:,x)./fac;
end

[coeff,scores,latent] = pca(heart2);
scores = scores';
coeff = coeff';

group1 = scores(:,1:160);
group2 = scores(:,161:297);
% group2 = scores(:,161:214);
% group3 = scores(:,215:249);
% group4 = scores(:,250:284);
group5 = scores(:,285:297);
figure
plot(group1(1,:), group1(2,:), 'b.')
hold on 
plot(group5(1,:), group5(2,:), 'r.');
hold off;
figure;


% plot3(group1(1,:),group1(2,:),group1(3,:),'b.',group2(1,:),...
%     group2(2,:),group2(3,:),'r.',group3(1,:),group3(2,:),group3(3,:),'g.',...
%     group4(1,:),group4(2,:),group4(3,:),'c.',group5(1,:),group5(2,:),group5(3,:),'m.');

plot3(group1(1,:),group1(2,:),group1(3,:),'b.',group5(1,:),group5(2,:),group5(3,:),'r.');


% 
% x = scores(:,1);
% y = scores(:,2);
% z = scores(:,3);
% 
% gscatter3(x,y,z);