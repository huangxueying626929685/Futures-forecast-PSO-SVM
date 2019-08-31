% 支持向量机用于收盘价预测，首先是未优化的，其次是优化后的
%% 清空环境
tic;clc;clear;close all;format compact
%% 加载数据
% data=xlsread('EUA五分钟.xlsx','G2:G30763'); save data data
load data
% 归一化
[a,inputns]=mapminmax(data',0,1);%归一化函数要求输入为行向量
data_trans=data_process(5,a);%% 对时间序列预测建立滚动序列，即用1到m个数据预测第m+1个数据，然后用2到m+1个数据预测第m+2个数据
input=data_trans(:,1:end-1);
output=data_trans(:,end);
%% 数据集 前75%训练 后25%预测
m=round(size(data_trans,1)*0.75);
Pn_train=input(1:m,:);
Tn_train=output(1:m,:);
Pn_test=input(m+1:end,:);
Tn_test=output(m+1:end,:);

%% 1.没有优化的SVM
bestc=0.001;bestg=10;%c和g随机赋值 表示没有优化的SVM
t=0;%t=0为线性核函数,1-多项式。2rbf核函数
cmd = ['-s 3 -t ',num2str(t),' -c ', num2str(bestc),' -g ',num2str(bestg),' -p 0.01 -d 1'];  
    
model = svmtrain(Tn_train,Pn_train,cmd);%训练
[predict,~]= svmpredict(Tn_test,Pn_test,model);%测试
% 反归一化，为后面的结果计算做准备
predict0=mapminmax('reverse',predict',inputns);%测试实际输出反归一化
T_test=mapminmax('reverse',Tn_test',inputns);%测试集期望输出反归一化
T_train=mapminmax('reverse',Tn_train',inputns);%训练集期望输出反归一化

figure
plot(predict0,'r-')
hold on;grid on
plot(T_test,'b-')
xlabel('样本编号')
ylabel('收盘价/元')
if t==0
    title('线性核SVM预测')
elseif t==1
    title('多项式核SVM预测')
else
    title('RBF核SVM预测')
end
legend('实际输出','期望输出')

figure
error_svm=abs(predict0-T_test)./T_test*100;%测试集每个样本的相对误差
plot(error_svm,'r-*')
xlabel('样本编号')
ylabel('收盘价相对误差/%')
if t==0
    title('线性核SVM预测的误差')
elseif t==1
    title('多项式核SVM预测的误差')
else
    title('RBF核SVM预测的误差')
end
grid on



%% 2.设计PSO优化SVM，用于选择最佳的C和G
pso_option = struct('c1',1,'c2',1,'maxgen',20,'sizepop',5,'k',0.6,'wV',0.9,'wP',0.9, ...
    'popcmax',10^1,'popcmin',10^(-3),'popgmax',10^1,'popgmin',10^(-3),'popkernel',t);%参数的解释在psoSVMcgForRegress里面

[bestmse,bestc,bestg,trace] = psoSVMcgForRegress(Tn_train,Pn_train,Tn_test,Pn_test,pso_option);

figure;
plot(trace,'r-');
xlabel('进化代数');
ylabel('适应度值(均方差)');
title('适应度曲线')
grid on;

% 利用PSO优化得到的最优参数进行SVM 重新训练
cmd = ['-s 3 -t ',num2str(t)',' -c ', num2str(bestc), ' -g ', num2str(bestg),' -p 0.01 -d 1'];
model = svmtrain(Tn_train,Pn_train,cmd);%训练
[predict_train,~]= svmpredict(Tn_train,Pn_train,model);%训练集
[predict,fit]= svmpredict(Tn_test,Pn_test,model);%测试集
% 反归一化
predict_tr=mapminmax('reverse',predict_train',inputns);%训练集实际输出反归一化
predict1=mapminmax('reverse',predict',inputns);%测试集输出反归一化

figure
plot(predict1,'r-')
hold on;grid on
plot(T_test,'b-')
xlabel('测试集样本编号')
ylabel('收盘价/元')
if t==0
    title('PSO-线性核SVM预测')
elseif t==1
    title('PSO-多项式核SVM预测')
else
    title('PSO-RBF核SVM预测')
end
legend('实际输出','期望输出')


figure
error_pso_svm=abs(predict1-T_test)./T_test*100;%测试集每个样本的相对误差
plot(error_pso_svm,'r-*')
xlabel('测试集样本编号')
ylabel('收盘价相对误差/%')
if t==0
    title('PSO-线性核SVM预测的误差')
elseif t==1
    title('PSO-多项式核SVM预测的误差')
else
    title('PSO-RBF核SVM预测的误差')
end
grid on

%% 结果分析

disp('最优惩罚参数与核参数为：')
bestc
bestg



disp('优化前的均方误差')
mse_svm=mse(predict0-T_test)
disp('优化前的平均相对误差')
mre_svm=sum(abs(predict0-T_test)./T_test)/length(T_test)
disp('优化前的平均绝对误差')
abs_svm=mean(abs(predict0-T_test))
disp('优化前的归一化均方误差')
a=sum((predict0-T_test).^2)/length(T_test);
b=sum((predict0-mean(predict0)).^2)/(length(T_test)-1);
one_svm=a/b

disp('优化后的训练集均方根误差')
rmse_svm0=sqrt(mse(predict_tr-T_train))
disp('优化后的测试集均方根误差')
rmse_svm=sqrt(mse(predict1-T_test))
disp('优化后的均方误差')
mse_pso_svm=mse(predict1-T_test)
disp('优化后的平均相对误差')
mre_pso_svm=sum(abs(predict1-T_test)./T_test)/length(T_test)
disp('优化后的平均绝对误差')
abs_pso_svm=mean(abs(predict1-T_test))
disp('优化后的归一化均方误差')
a1=sum((predict1-T_test).^2)/length(T_test);
b1=sum((predict1-mean(predict1)).^2)/(length(T_test)-1);
one_pso_svm=a1/b1


toc %结束计时