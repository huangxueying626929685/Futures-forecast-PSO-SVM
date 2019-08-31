function [bestCVmse,bestc,bestg,fit_gen] = psoSVMcgForRegress(train_label,train,T_test,P_test,pso_option)

%% 参数初始化
% c1:pso参数局部搜索能力
% c2:pso参数全局搜索能力
% maxgen:最大进化数量
% sizepop:种群最大数量
% k:k belongs to [0.1,1.0],速率和x的关系(V = kX)
% wV:(wV best belongs to [0.8,1.2]),速率更新公式中速度前面的弹性系数
% wP:种群更新公式中速度前面的弹性系数

% popcmax:SVM 参数c的变化的最大值.
% popcmin:SVM 参数c的变化的最小值.
% popgmax:SVM 参数g的变化的最大值.
% popgmin:SVM 参数c的变化的最小值.
% popkernel:SVM的核参数

Vcmax = pso_option.k*pso_option.popcmax;
Vcmin = -Vcmax ;
Vgmax = pso_option.k*pso_option.popgmax;
Vgmin = -Vgmax ;
%% 产生初始粒子和速度
for i=1:pso_option.sizepop
    % 随机产生种群和速度
    i
    pop(i,1) = (pso_option.popcmax-pso_option.popcmin)*rand+pso_option.popcmin;
    pop(i,2) = (pso_option.popgmax-pso_option.popgmin)*rand+pso_option.popgmin;
    V(i,1)=Vcmax*rands(1,1);
    V(i,2)=Vgmax*rands(1,1);
    
    % 计算初始适应度
    cmd = ['-s 3 -t ',num2str( pso_option.popkernel ),' -c ',num2str( pop(i,1) ),' -g ',num2str( pop(i,2) ),' -p 0.01 -d 1'];
    model= svmtrain(train_label, train, cmd);
    [l,~]= svmpredict(T_test,P_test,model);
    fitness(i)=mse(l-T_test);%以均方差作为适应度函数，均方差越小，精度越高
end

% 找极值和极值点
[global_fitness bestindex]=min(fitness); % 全局极值
local_fitness=fitness;   % 个体极值初始化

global_x=pop(bestindex,:);   % 全局极值点
local_x=pop;    % 个体极值点初始化

% 每一代种群的平均适应度
avgfitness_gen = zeros(1,pso_option.maxgen);

%% 迭代寻优
for i=1:pso_option.maxgen
    iter=i
    for j=1:pso_option.sizepop
        
        %速度更新
        V(j,:) = pso_option.wV*V(j,:) + pso_option.c1*rand*(local_x(j,:) - pop(j,:)) + pso_option.c2*rand*(global_x - pop(j,:));
        % 边界判断
        if V(j,1) > Vcmax
            V(j,1) = Vcmax;
        end
        if V(j,1) < Vcmin
            V(j,1) = Vcmin;
        end
        if V(j,2) > Vgmax
            V(j,2) = Vgmax;
        end
        if V(j,2) < Vgmin
            V(j,2) = Vgmin;
        end
        
        %种群更新
        pop(j,:)=pop(j,:) + pso_option.wP*V(j,:);
        %边界判断
        if pop(j,1) > pso_option.popcmax
            pop(j,1) = (pso_option.popcmax-pso_option.popcmin)*rand+pso_option.popcmin;;
        end
        if pop(j,1) < pso_option.popcmin
            pop(j,1) = (pso_option.popcmax-pso_option.popcmin)*rand+pso_option.popcmin;;
        end
        if pop(j,2) > pso_option.popgmax
            pop(j,2) = (pso_option.popgmax-pso_option.popgmin)*rand+pso_option.popgmin;;
        end
        if pop(j,2) < pso_option.popgmin
            pop(j,2) = (pso_option.popgmax-pso_option.popgmin)*rand+pso_option.popgmin;;
        end
        
        % 自适应粒子变异
        if rand>0.8
            k=ceil(2*rand);
            if k == 1
                pop(j,k) = (pso_option.popcmax-pso_option.popcmin)*rand + pso_option.popcmin;
            end
            if k == 2
                pop(j,k) = (pso_option.popgmax-pso_option.popgmin)*rand + pso_option.popgmin;
            end
        end
        
        %适应度值
        cmd = ['-t ',num2str( pso_option.popkernel ),' -c ',num2str( pop(j,1) ),' -g ',num2str( pop(j,2) ),' -s 3 -p 0.01 -d 1'];
        model= svmtrain(train_label, train, cmd);
        [l,mse1]= svmpredict(T_test,P_test,model);
        fitness(j)=mse(l-T_test);
        %个体最优更新
        if fitness(j) < local_fitness(j)
            local_x(j,:) = pop(j,:);
            local_fitness(j) = fitness(j);
        end
        
        if fitness(j) == local_fitness(j) && pop(j,1) < local_x(j,1)
            local_x(j,:) = pop(j,:);
            local_fitness(j) = fitness(j);
        end
        
        %群体最优更新
        if fitness(j) < global_fitness
            global_x = pop(j,:);
            global_fitness = fitness(j);
        end
    end
    
    fit_gen(i)=global_fitness;
    avgfitness_gen(i) = sum(fitness)/pso_option.sizepop;
end

%% 输出结果
% 最好的参数
bestc = global_x(1);
bestg = global_x(2);
bestCVmse = fit_gen(pso_option.maxgen);%最好的结果

