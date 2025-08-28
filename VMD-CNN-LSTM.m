clc;clear;close all;	
load('MT_28_Aug_2025_15_50_47.mat')	
random_seed=G_out_data.random_seed;	
rng(random_seed)	
	
data_str=G_out_data.data_path_str ;  %读取数据的路径 	
dataO1=readtable(data_str,'VariableNamingRule','preserve'); %读取数据 	
data1=dataO1(:,2:end);test_data=table2cell(dataO1(1,2:end));	
for i=1:length(test_data)	
      if ischar(test_data{1,i})==1	
          index_la(i)=1;     %char类型	
      elseif isnumeric(test_data{1,i})==1	
          index_la(i)=2;     %double类型	
      else	
        index_la(i)=0;     %其他类型	
    end 	
end	
index_char=find(index_la==1);index_double=find(index_la==2);	
	
%%数值类型数据处理	
if length(index_double)>=1	
     data_numshuju=table2array(data1(:,index_double));	
     data_numshuju2=data_numshuju;	
     index_need_last=index_double;	
else	
     index_need_last=index_double;	
    data_numshuju2=[];	
end	
	
% 文本类型数据处理	
data_shuju=[];	
if length(index_char)>=1	
   for j=1:length(index_char)	
     data_get=table2array(data1(:,index_char(j)));	
	
     data_label=unique(data_get);	
     for NN=1:length(data_label)	
         idx = find(ismember(data_get,data_label{NN,1}));	
         data_shuju(idx,j)=NN;	
     end	
   end	
end	
data_all_last=[data_shuju,data_numshuju2];	
label_all_last=[index_char,index_need_last];	
data=data_all_last;	
     data_biao_all=data1.Properties.VariableNames;	
for j=1:length(label_all_last)	
     data_biao{1,j}=data_biao_all{1,label_all_last(j)};	
end	
	
	
	
	
data_numshuju=data;	
for NN=1:size(data_numshuju,2)	
      data_test=data_numshuju(:,NN);	
      index=isnan(data_test);	
      data_test1=data_test;	
      data_test1(index)=[];	
      index_label=1:length(data_test);	
      index_label1=index_label;	
      index_label1(index)=[];	
     data_all=interp1(index_label1,data_test1,index_label,'spline');	
	
     dataO(:,NN)=data_all;	
end	
	
A_data1=dataO;	
data_biao1=data_biao;	
select_feature_num=G_out_data.select_feature_num;   %特征选择的个数	
predict_num=G_out_data.predict_num_set;   %预测的点个数	
	
data_select=A_data1;	
feature_need_last=1:size(A_data1,2)-predict_num;	
	
data_select1=data_select;	
data_select1=data_select1;	
	
	
	
%%波形分解	
data_select1_cell=[];	
t=1:length(data_select1(:,end));	
deo_num=G_out_data.deo_num;	
figure	
[imf,res] = vmd(data_select1(:,end),'NumIMF',deo_num);	
[p,q] = ndgrid(t,1:size(imf,2));	
imf_L=[data_select1(:,end),imf,res];	
P=[p,p(:,1:2)];	
Q=[q(:,1),q+1,q(:,end)+2];	
plot3(P,Q,imf_L)	
grid on	
xlabel('Time Values')	
ylabel('Mode Number')	
zlabel('Mode Amplitude')	
	
decom_str{1,1}='origin data';	
for i=1:size(imf,2)	
     decom_str{1,i+1}=['imf',num2str(i)];	
end	
decom_str{1,2+size(imf,2)}='res';	
yticks(1:length(decom_str))	
yticklabels(decom_str)	
	
for NN1=1:deo_num	
    data_select1_cell{1,NN1}=[data_select1(:,1:end-1),imf(:,NN1)];	
end	
	
data_select1_cell{1,NN1+1}=[data_select1(:,1:end-1),res];	
	
plotpl( data_select1(:,end),[imf,res]')	
	
	
 % 模型训练参数	
select_predict_num=G_out_data.select_predict_num;  %待预测数	
num_feature=G_out_data.num_feature;    %特征选择量	
num_series=G_out_data.num_series;     %序列选择	
num_input_serise=num_series;     	
min_batchsize=G_out_data.min_batchsize; 	
roll_num=G_out_data.roll_num;    %滚动次数	
roll_num_in=G_out_data.roll_num_in;	
num_pop=G_out_data.num_pop;  %优化种群数	
num_iter=G_out_data.num_iter;  %优化迭代数	
num_BO_iter=G_out_data.num_BO_iter;  %贝叶斯优化迭代次数	
max_epoch_LC=G_out_data.max_epoch_LC; %最大轮数	
method_mti=G_out_data.method_mti; %最大轮数	
list_cell=	G_out_data.list_cell;	
	
attention_label=G_out_data.attention_label;	
attention_head=G_out_data.attention_head;	
	
%% 模型训练	
x_mu_all=[];x_sig_all=[];y_mu_all=[];y_sig_all=[];  	
for NUM_all=1:length(data_select1_cell)	
    data_process=data_select1_cell{1,NUM_all};	
   [x_feature_label,y_feature_label]=timeseries_process1(data_process,select_predict_num,num_feature,num_series);  	
   [~,y_feature_label1]=timeseries_process1(data_select1,select_predict_num,num_feature,num_series);    %未分解之前	
	
	
	
  index_label1=1:(size(x_feature_label,1)); index_label=index_label1;	
  spilt_ri=G_out_data.spilt_ri;	
  train_num=round(spilt_ri(1)/(sum(spilt_ri))*size(x_feature_label,1));                    %训练集个数	
  vaild_num=round((spilt_ri(1)+spilt_ri(2))/(sum(spilt_ri))*size(x_feature_label,1)); %验证集个数	
  %训练集，验证集，测试集	
  train_x_feature_label=x_feature_label(index_label(1:train_num),:);	
  train_y_feature_label=y_feature_label(index_label(1:train_num),:);	
  vaild_x_feature_label=x_feature_label(index_label(train_num+1:vaild_num),:);	
  vaild_y_feature_label=y_feature_label(index_label(train_num+1:vaild_num),:);	
  test_x_feature_label=x_feature_label(index_label(vaild_num+1:end),:);	
  test_y_feature_label=y_feature_label(index_label(vaild_num+1:end),:);	
  %Zscore 标准化	
	
  %训练集	
  x_mu = mean(train_x_feature_label);  x_sig = std(train_x_feature_label); 	
  train_x_feature_label_norm = (train_x_feature_label - x_mu) ./ x_sig;    % 训练数据标准化	
  y_mu = mean(train_y_feature_label);  y_sig = std(train_y_feature_label); 	
  train_y_feature_label_norm = (train_y_feature_label - y_mu) ./ y_sig;    % 训练数据标准化	
  x_mu_all(NUM_all,:)=x_mu;x_sig_all(NUM_all,:)=x_sig;y_mu_all(NUM_all,:)=y_mu;y_sig_all(NUM_all,:)=y_sig;                   	
  %验证集	
  vaild_x_feature_label_norm = (vaild_x_feature_label - x_mu) ./ x_sig;    % 训练数据标准化	
  vaild_y_feature_label_norm = (vaild_y_feature_label - y_mu) ./ y_sig;    % 训练数据标准化	
  %测试集	
  test_x_feature_label_norm = (test_x_feature_label - x_mu) ./ x_sig;    % 训练数据标准化	
  test_y_feature_label_norm = (test_y_feature_label - y_mu) ./ y_sig;    % 训练数据标准化	
	
	
	
	
y_train_predict_norm=zeros(size(train_y_feature_label,1),size(train_y_feature_label,2));y_vaild_predict_norm=zeros(size(vaild_y_feature_label,1),size(vaild_y_feature_label,2));	
y_test_predict_norm=zeros(size(test_y_feature_label,1),size(test_y_feature_label,2));	
	
  for N1=1:length(list_cell)	
    hidden_size=G_out_data.hidden_size;	
    p_train1=reshape(train_x_feature_label_norm',size(train_x_feature_label_norm,2),1,1,size(train_x_feature_label_norm,1));	
    	
    p_vaild1=reshape(vaild_x_feature_label_norm',size(vaild_x_feature_label,2),1,1,size(vaild_x_feature_label,1));	
   	
  p_test1=reshape(test_x_feature_label_norm',size(test_x_feature_label,2),1,1,size(test_x_feature_label,1));	
 	
	
	
       layers = [imageInputLayer([ size(train_x_feature_label,2) 1 1])%%2D-CNN   ) 	
        convolution2dLayer([2,1],10)	
        batchNormalizationLayer                                           	
	
       reluLayer	
       maxPooling2dLayer([2 1],'Stride',1) 	
       flattenLayer    	
       lstmLayer(hidden_size(1), 'OutputMode', 'sequence')      % LSTM层	
       reluLayer	
       fullyConnectedLayer(size(train_y_feature_label(:,list_cell{1,N1}),2))	
       regressionLayer];	
	
      options = trainingOptions('adam', ...	
         'MaxEpochs',max_epoch_LC, ...	
         'MiniBatchSize',min_batchsize,...	
         'InitialLearnRate',0.001,...	
         'ValidationFrequency',20, ...	
         'LearnRateSchedule','piecewise', ...	
         'LearnRateDropPeriod',125, ...	
        'LearnRateDropFactor',0.2, ...	
        'Plots','training-progress');	
	
	
   [Mdl,Loss] = trainNetwork(p_train1,  train_y_feature_label_norm(:,list_cell{1,N1}), layers, options);	
    y_train_predict_norm_roll = predict(Mdl, p_train1,'MiniBatchSize',min_batchsize);	
    y_vaild_predict_norm_roll = predict(Mdl, p_vaild1,'MiniBatchSize',min_batchsize);	
    y_test_predict_norm_roll =  predict(Mdl, p_test1,'MiniBatchSize',min_batchsize);	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
y_train_predict_norm(:,list_cell{1,N1})=y_train_predict_norm_roll;	
y_vaild_predict_norm(:,list_cell{1,N1})=y_vaild_predict_norm_roll;	
y_test_predict_norm(:,list_cell{1,N1})=y_test_predict_norm_roll;	
 lgraph = layerGraph(Mdl.Layers);	
  figure 	
 plot(lgraph)	
 model_all{NUM_all,N1}=Mdl;	
   	
	
	
	
figure	
	
subplot(2, 1, 1)	
plot(1 : length(Loss.TrainingRMSE), Loss.TrainingRMSE, '-', 'LineWidth', 1)	
xlabel('迭代次数');ylabel('均方根误差');legend('训练集均方根误差');title ('训练集均方根误差曲线');grid;set(gcf,'color','w')	
	
subplot(2, 1, 2)	
plot(1 : length(Loss.TrainingLoss), Loss.TrainingLoss, '-', 'LineWidth', 1)	
xlabel('迭代次数');ylabel('损失函数');legend('训练集损失值');title ('训练集损失函数曲线');grid;set(gcf,'color','w')	
	
  end	
	
	
y_train_predict_cell{1,NUM_all}=y_train_predict_norm.*y_sig+y_mu;  %反标准化操作	
y_vaild_predict_cell{1,NUM_all}=y_vaild_predict_norm.*y_sig+y_mu;	
 y_test_predict_cell{1,NUM_all}=y_test_predict_norm.*y_sig+y_mu;	
	
end	
	
	
	
y_train_predict=0;y_vaild_predict=0;y_test_predict=0;	
for i=1:length(data_select1_cell)	
      y_train_predict=y_train_predict+ y_train_predict_cell{1,i};	
      y_vaild_predict=y_vaild_predict+ y_vaild_predict_cell{1,i};	
      y_test_predict=y_test_predict+ y_test_predict_cell{1,i};	
end	
	
train_y_feature_label=y_feature_label1(index_label(1:train_num),:); 	
vaild_y_feature_label=y_feature_label1(index_label(train_num+1:vaild_num),:);	
test_y_feature_label=y_feature_label1(index_label(vaild_num+1:end),:);	
	
Tvalue=G_out_data.Tvalue;  %使用的方法	
	
train_y=train_y_feature_label; 	
train_MAE=sum(sum(abs(y_train_predict-train_y)))/size(train_y,1)/size(train_y,2) ; disp([Tvalue,'训练集平均绝对误差MAE：',num2str(train_MAE)])	
train_MAPE=sum(sum(abs((y_train_predict-train_y)./train_y)))/size(train_y,1)/size(train_y,2); disp([Tvalue,'训练集平均相对误差MAPE：',num2str(train_MAPE)])	
train_MSE=(sum(sum(((y_train_predict-train_y)).^2))/size(train_y,1)/size(train_y,2)); disp([Tvalue,'训练集均方误差MSE：',num2str(train_MSE)])    	
train_RMSE=sqrt(sum(sum(((y_train_predict-train_y)).^2))/size(train_y,1)/size(train_y,2)); disp([Tvalue,'训练集均方根误差RMSE：',num2str(train_RMSE)]) 	
train_R2 = 1 - mean(norm(train_y - y_train_predict)^2 / norm(train_y - mean(train_y))^2);   disp([Tvalue,'训练集R方系数R2：',num2str(train_R2)]) 	
disp('************************************************************************************')	
vaild_y=vaild_y_feature_label;	
vaild_MAE=sum(sum(abs(y_vaild_predict-vaild_y)))/size(vaild_y,1)/size(vaild_y,2) ; disp([Tvalue,'验证集平均绝对误差MAE：',num2str(vaild_MAE)])	
vaild_MAPE=sum(sum(abs((y_vaild_predict-vaild_y)./vaild_y)))/size(vaild_y,1)/size(vaild_y,2); disp([Tvalue,'验证集平均相对误差MAPE：',num2str(vaild_MAPE)])	
vaild_MSE=(sum(sum(((y_vaild_predict-vaild_y)).^2))/size(vaild_y,1)/size(vaild_y,2)); disp([Tvalue,'验证集均方误差MSE：',num2str(vaild_MSE)])     	
vaild_RMSE=sqrt(sum(sum(((y_vaild_predict-vaild_y)).^2))/size(vaild_y,1)/size(vaild_y,2)); disp([Tvalue,'验证集均方根误差RMSE：',num2str(vaild_RMSE)]) 	
vaild_R2 = 1 - mean(norm(vaild_y - y_vaild_predict)^2 / norm(vaild_y - mean(vaild_y))^2);   disp([Tvalue,'验证集R方系数R2：',num2str(vaild_R2)]) 	
disp('************************************************************************************')	
test_y=test_y_feature_label;	
test_MAE=sum(sum(abs(y_test_predict-test_y)))/size(test_y,1)/size(test_y,2) ; disp([Tvalue,'测试集平均绝对误差MAE：',num2str(test_MAE)])	
test_MAPE=sum(sum(abs((y_test_predict-test_y)./test_y)))/size(test_y,1)/size(test_y,2); disp([Tvalue,'测试集平均相对误差MAPE：',num2str(test_MAPE)])	
test_MSE=(sum(sum(((y_test_predict-test_y)).^2))/size(test_y,1)/size(test_y,2)); disp([Tvalue,'测试集均方误差MSE：',num2str(test_MSE)]) 	
test_RMSE=sqrt(sum(sum(((y_test_predict-test_y)).^2))/size(test_y,1)/size(test_y,2)); disp([Tvalue,'测试集均方根误差RMSE：',num2str(test_RMSE)]) 	
test_R2 = 1 - mean(norm(test_y - y_test_predict)^2 / norm(test_y - mean(test_y))^2);   disp([Tvalue,'测试集R方系数R2：',num2str(test_R2)]) 	
	
	
	
	
%% 调用模型得到数据预测结果	
	
y_last_predict_cell=cell(1,length(data_select1_cell));	
for NUM_all=1:length(data_select1_cell)	
      data_process=data_select1_cell{1,NUM_all};	
      data_process=data_process(vaild_num+1:end,:);	
	
       [x_feature_label]=timeseries_process1_Pre(data_process,select_predict_num,num_feature,num_series);	
	
       x_mu=x_mu_all(NUM_all,:);	
       x_sig=x_sig_all(NUM_all,:);	
       pre_x_feature_label_norm = (x_feature_label - x_mu) ./ x_sig;    % 训练数据标准化	
	
      	
      p_pre1=reshape(pre_x_feature_label_norm',size(pre_x_feature_label_norm,2),1,1,size(pre_x_feature_label_norm,1));	
   	
   	
       for N1=1:length(list_cell)	
            Mdl=model_all{NUM_all,N1};	
            y_pre_predict_norm1 =  predict(Mdl, p_pre1,'MiniBatchSize',min_batchsize);	
            y_pre_predict_norm(:,list_cell{1,N1})=y_pre_predict_norm1;	
       end	
	
    	
     	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
        y_last_predict_cell{1,NUM_all}=y_pre_predict_norm.*y_sig_all(NUM_all,:)+y_mu_all(NUM_all,:);  %反标准化操作	
    end	
	
y_last_predict=0;   y_before=0;	
for i=1:length(data_select1_cell)	
      y_last_predict=y_last_predict+ y_last_predict_cell{1,i};	
      y_before=y_before+data_select1_cell{1,i}(vaild_num+1:end,end);	
end	
	
y_last_predict1=y_last_predict(end,1:end);	
disp('预测未来时间点数据为:')	
	
disp(y_last_predict1)	
figure;	
plot([1:length(y_before)],y_before,'-o','LineWidth',1);		
hold on	
plot([length(y_before)+1:length(y_before)+length(y_last_predict1)],y_last_predict1,'-p','LineWidth',1.2)		
hold on	
legend('True','Predict')	
set(gca,'LineWidth',1.2)	
