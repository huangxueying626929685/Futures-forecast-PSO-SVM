function [train_data_norm]=data_process(numdely,a)
if size(a,1)==1
    a=a';
end

numdata = size(a,1);

numsample = numdata - numdely - 1;
train_data_norm = zeros(numdely+1, numsample);
for i = 1 :numsample
    train_data_norm(:,i) = a(i:i+numdely)';
end     
train_data_norm=train_data_norm';

