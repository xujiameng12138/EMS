clc; clear; close all;

%% =========================
%  Load configuration / meta
% ==========================
load('DATE.mat')

random_seed = G_out_data.random_seed;
rng(random_seed);

%% =========================
%  Read data
% ==========================
data_str = G_out_data.data_path_str;  % Path to input table
dataO1   = readtable(data_str, 'VariableNamingRule','preserve');
data1    = dataO1(:, 2:end);

% Detect variable types from first row
test_data = table2cell(dataO1(1, 2:end));
index_la  = zeros(1, length(test_data));
for i = 1:length(test_data)
    if ischar(test_data{1,i})
        index_la(i) = 1;     % char
    elseif isnumeric(test_data{1,i})
        index_la(i) = 2;     % numeric
    else
        index_la(i) = 0;     % other
    end
end
index_char   = find(index_la == 1);
index_double = find(index_la == 2);

%% =========================
%  Process numeric columns
% ==========================
if ~isempty(index_double)
    data_num = table2array(data1(:, index_double));
    index_need_last = index_double;
else
    data_num = [];
    index_need_last = index_double;
end

%% =========================
%  Process text columns -> ordinal encoding
% ==========================
data_txt = [];
if ~isempty(index_char)
    data_txt = zeros(height(data1), length(index_char));
    for j = 1:length(index_char)
        data_get   = table2array(data1(:, index_char(j)));
        data_label = unique(data_get);
        for NN = 1:length(data_label)
            idx = find(ismember(data_get, data_label{NN,1}));
            data_txt(idx, j) = NN;
        end
    end
end

% Combine (text-coded + numeric)
data_all_last   = [data_txt, data_num];
label_all_last  = [index_char, index_need_last];
data            = data_all_last;

% Keep variable names for selected columns
data_biao_all = data1.Properties.VariableNames;
data_biao     = cell(1, length(label_all_last));
for j = 1:length(label_all_last)
    data_biao{1,j} = data_biao_all{1, label_all_last(j)};
end

%% ==========================================================
%  Causal missing-value handling (history-only imputation)
%  Requirement: any imputation must use ONLY past observations
% ==========================================================
data_numshuju = data;
dataO = zeros(size(data_numshuju));

for NN = 1:size(data_numshuju, 2)
    x = data_numshuju(:, NN);

    % 1) Forward-fill using only historical observations (causal)
    x_ff = fillmissing(x, 'previous');

    % 2) Handle leading NaNs (no historical observation exists)
    if any(isnan(x_ff))
        firstValid = find(~isnan(x_ff), 1, 'first');
        if ~isempty(firstValid)
            x_ff(1:firstValid-1) = x_ff(firstValid); % constant backfill at head
        end
    end

    % 3) If still all NaN (degenerate), set to zeros (or throw error)
    if all(isnan(x_ff))
        x_ff = zeros(size(x_ff));
    end

    dataO(:, NN) = x_ff;
end

A_data1     = dataO;
data_biao1  = data_biao;

select_feature_num = G_out_data.select_feature_num;   %#ok<NASGU> % number of selected features
predict_num        = G_out_data.predict_num_set;      % number of prediction horizons (output dims)

data_select   = A_data1;
feature_need_last = 1:size(A_data1,2)-predict_num;    %#ok<NASGU>

data_select1  = data_select;

%% =========================
%  Optimized VMD decomposition
%  - Fix K = deo_num
%  - Optimize alpha via population-based metaheuristic (PSO)
%  - Objective: minimize reconstruction RMSE
% ==========================
t = 1:length(data_select1(:, end));
deo_num = G_out_data.deo_num;     % number of IMFs (K)

signal = data_select1(:, end);
K      = deo_num;

% Search range for alpha (penalty on bandwidth)
alpha_lb = 1e2;
alpha_ub = 1e6;

% Use predefined population size and iterations (for stable convergence & cost control)
popSize  = G_out_data.num_pop;
maxIters = G_out_data.num_iter;

% Optimize alpha
rng(random_seed);
[alpha_opt, best_rmse] = optimize_vmd_alpha_pso(signal, K, alpha_lb, alpha_ub, popSize, maxIters);

% Run VMD with optimized alpha
[imf, res] = vmd(signal, 'NumIMF', K, 'Alpha', alpha_opt);

disp(['[Optimized VMD] alpha* = ', num2str(alpha_opt), ...
      ', reconstruction RMSE = ', num2str(best_rmse)]);

% Optional visualization (3D mode plot)
figure;
[p,q] = ndgrid(t, 1:size(imf,2));
imf_L = [signal, imf, res];
P = [p, p(:,1:2)];
Q = [q(:,1), q+1, q(:,end)+2];
plot3(P, Q, imf_L);
grid on;
xlabel('Time Index');
ylabel('Mode Index');
zlabel('Mode Amplitude');

decom_str = cell(1, size(imf,2)+2);
decom_str{1,1} = 'Original';
for i = 1:size(imf,2)
    decom_str{1,i+1} = ['IMF', num2str(i)];
end
decom_str{1,2+size(imf,2)} = 'Residual';
yticks(1:length(decom_str));
yticklabels(decom_str);

% Prepare decomposed datasets: each IMF (and residual) becomes a target column
data_select1_cell = cell(1, K+1);
for NN1 = 1:K
    data_select1_cell{1,NN1} = [data_select1(:,1:end-1), imf(:,NN1)];
end
data_select1_cell{1,K+1} = [data_select1(:,1:end-1), res];

% Keep your original helper plot call (assumed available in your project)
plotpl(signal, [imf,res]');

%% =========================
%  Model training parameters
% ==========================
select_predict_num = G_out_data.select_predict_num;  % # points to predict
num_feature        = G_out_data.num_feature;         % # selected features
num_series         = G_out_data.num_series;          % # lags / series length
num_input_serise   = num_series;                     %#ok<NASGU>

min_batchsize      = G_out_data.min_batchsize;
roll_num           = G_out_data.roll_num;            %#ok<NASGU>
roll_num_in        = G_out_data.roll_num_in;

num_pop            = G_out_data.num_pop;             %#ok<NASGU>
num_iter           = G_out_data.num_iter;            %#ok<NASGU>
num_BO_iter        = G_out_data.num_BO_iter;
max_epoch_LC       = G_out_data.max_epoch_LC;
method_mti         = G_out_data.method_mti;          %#ok<NASGU>
list_cell          = G_out_data.list_cell;

attention_label    = G_out_data.attention_label;
attention_head     = G_out_data.attention_head;

%% =========================
%  Train per component and reconstruct prediction
% ==========================
x_mu_all = []; x_sig_all = []; y_mu_all = []; y_sig_all = [];
y_train_predict_cell = cell(1, length(data_select1_cell));
y_vaild_predict_cell = cell(1, length(data_select1_cell));
y_test_predict_cell  = cell(1, length(data_select1_cell));

model_all = cell(length(data_select1_cell), length(list_cell));

for NUM_all = 1:length(data_select1_cell)

    data_process = data_select1_cell{1, NUM_all};

    [x_feature_label, y_feature_label] = timeseries_process(data_process, select_predict_num, num_feature, num_series);
    [~, y_feature_label1] = timeseries_process(data_select1,  select_predict_num, num_feature, num_series); % original (non-decomposed) targets

    index_label1 = 1:size(x_feature_label,1);
    index_label  = index_label1;

    spilt_ri = G_out_data.spilt_ri;
    train_num = round(spilt_ri(1)/sum(spilt_ri) * size(x_feature_label,1));
    vaild_num = round((spilt_ri(1)+spilt_ri(2))/sum(spilt_ri) * size(x_feature_label,1));

    train_x_feature_label = x_feature_label(index_label(1:train_num), :);
    train_y_feature_label = y_feature_label(index_label(1:train_num), :);

    vaild_x_feature_label = x_feature_label(index_label(train_num+1:vaild_num), :);
    vaild_y_feature_label = y_feature_label(index_label(train_num+1:vaild_num), :);

    test_x_feature_label  = x_feature_label(index_label(vaild_num+1:end), :);
    test_y_feature_label  = y_feature_label(index_label(vaild_num+1:end), :);

    % Z-score normalization based on training set
    x_mu  = mean(train_x_feature_label);
    x_sig = std(train_x_feature_label);
    train_x_feature_label_norm = (train_x_feature_label - x_mu) ./ x_sig;

    y_mu  = mean(train_y_feature_label);
    y_sig = std(train_y_feature_label);
    train_y_feature_label_norm = (train_y_feature_label - y_mu) ./ y_sig;

    x_mu_all(NUM_all,:) = x_mu;  x_sig_all(NUM_all,:) = x_sig;
    y_mu_all(NUM_all,:) = y_mu;  y_sig_all(NUM_all,:) = y_sig;

    vaild_x_feature_label_norm = (vaild_x_feature_label - x_mu) ./ x_sig;
    vaild_y_feature_label_norm = (vaild_y_feature_label - y_mu) ./ y_sig;

    test_x_feature_label_norm  = (test_x_feature_label  - x_mu) ./ x_sig;
    test_y_feature_label_norm  = (test_y_feature_label  - y_mu) ./ y_sig;

    y_train_predict_norm = zeros(size(train_y_feature_label,1), size(train_y_feature_label,2));
    y_vaild_predict_norm = zeros(size(vaild_y_feature_label,1), size(vaild_y_feature_label,2));
    y_test_predict_norm  = zeros(size(test_y_feature_label,1),  size(test_y_feature_label,2));

    for N1 = 1:length(list_cell)

        hidden_size = G_out_data.hidden_size; %#ok<NASGU>

        p_train1 = reshape(train_x_feature_label_norm', size(train_x_feature_label_norm,2), 1, 1, size(train_x_feature_label_norm,1));
        p_vaild1 = reshape(vaild_x_feature_label_norm', size(vaild_x_feature_label_norm,2), 1, 1, size(vaild_x_feature_label_norm,1));
        p_test1  = reshape(test_x_feature_label_norm',  size(test_x_feature_label_norm,2),  1, 1, size(test_x_feature_label_norm,1));

        data_struct1 = struct();

        opt = struct();
        opt.methods = 'CNN-LSTM';
        opt.maxEpochs = max_epoch_LC;
        opt.miniBatchSize = min_batchsize;
        opt.executionEnvironment = 'auto';   % 'cpu' 'gpu' 'auto'
        opt.LR = 'adam';                     % 'sgdm' 'rmsprop' 'adam'
        opt.trainingProgress = 'none';       % 'training-progress' 'none'
        opt.isUseBiLSTMLayer = true;
        opt.isUseDropoutLayer = true;
        opt.DropoutValue = 0.2;

        % -------- Optimization variables (Bayesian optimization)
        opt.optimVars = [
            optimizableVariable('NumOfLayer', [1 2], 'Type','integer')
            optimizableVariable('NumOfUnits', [50 200], 'Type','integer')
            optimizableVariable('isUseBiLSTMLayer', [1 2], 'Type','integer')
            optimizableVariable('InitialLearnRate', [1e-2 1], 'Transform','log')
            optimizableVariable('L2Regularization', [1e-10 1e-2], 'Transform','log')
        ];

        opt.isUseOptimizer = true;
        opt.isSaveOptimizedValue = false;
        opt.isSaveBestOptimizedValue = true;

        opt.MaxOptimizationTime   = 14*60*60;
        opt.MaxItrationNumber     = num_BO_iter;
        opt.isDispOptimizationLog = true;

        opt.roll_num_in = roll_num_in;

        data_struct1.X   = x_feature_label;
        data_struct1.Y   = y_feature_label;
        data_struct1.XTr = p_train1;
        data_struct1.YTr = train_y_feature_label_norm(:, list_cell{1,N1});
        data_struct1.XTs = p_test1;
        data_struct1.YTs = test_y_feature_label_norm(:,  list_cell{1,N1});
        data_struct1.XVl = p_vaild1;
        data_struct1.YVl = vaild_y_feature_label_norm(:, list_cell{1,N1});
        data_struct1.attention_label = attention_label;
        data_struct1.attention_head  = attention_head;

        % Bayesian optimization + train
        [opt, data_struct1] = OptimizeBaye_CNNS1(opt, data_struct1);
        [~, data_struct1, ~, Loss] = EvaluationData2(opt, data_struct1);

        Mdl = data_struct1.BiLSTM.Net;

        % Predict
        y_train_predict_norm_roll = predict(Mdl, p_train1, 'MiniBatchSize', opt.miniBatchSize);
        y_vaild_predict_norm_roll = predict(Mdl, p_vaild1, 'MiniBatchSize', opt.miniBatchSize);
        y_test_predict_norm_roll  = predict(Mdl, p_test1,  'MiniBatchSize', opt.miniBatchSize);

        y_train_predict_norm(:, list_cell{1,N1}) = y_train_predict_norm_roll;
        y_vaild_predict_norm(:, list_cell{1,N1}) = y_vaild_predict_norm_roll;
        y_test_predict_norm(:,  list_cell{1,N1}) = y_test_predict_norm_roll;

        % Plot network graph
        lgraph = layerGraph(Mdl.Layers);
        figure;
        plot(lgraph);

        model_all{NUM_all, N1} = Mdl;

        % Training curves
        figure;
        subplot(2,1,1);
        plot(1:length(Loss.TrainingRMSE), Loss.TrainingRMSE, '-', 'LineWidth', 1);
        xlabel('Iteration');
        ylabel('RMSE');
        legend('Training RMSE');
        title('Training RMSE Curve');
        grid on; set(gcf,'color','w');

        subplot(2,1,2);
        plot(1:length(Loss.TrainingLoss), Loss.TrainingLoss, '-', 'LineWidth', 1);
        xlabel('Iteration');
        ylabel('Loss');
        legend('Training Loss');
        title('Training Loss Curve');
        grid on; set(gcf,'color','w');

    end

    % De-normalize (component-level)
    y_train_predict_cell{1,NUM_all} = y_train_predict_norm .* y_sig + y_mu;
    y_vaild_predict_cell{1,NUM_all} = y_vaild_predict_norm .* y_sig + y_mu;
    y_test_predict_cell{1,NUM_all}  = y_test_predict_norm  .* y_sig + y_mu;

end

%% =========================
%  Reconstruct final predictions by summation over components
% ==========================
y_train_predict = 0;
y_vaild_predict = 0;
y_test_predict  = 0;

for i = 1:length(data_select1_cell)
    y_train_predict = y_train_predict + y_train_predict_cell{1,i};
    y_vaild_predict = y_vaild_predict + y_vaild_predict_cell{1,i};
    y_test_predict  = y_test_predict  + y_test_predict_cell{1,i};
end

% True targets (from original non-decomposed series)
train_y_feature_label = y_feature_label1(index_label(1:train_num), :);
vaild_y_feature_label = y_feature_label1(index_label(train_num+1:vaild_num), :);
test_y_feature_label  = y_feature_label1(index_label(vaild_num+1:end), :);

Tvalue = G_out_data.Tvalue;

%% =========================
%  Metrics (English output)
% ==========================
train_y = train_y_feature_label;
train_MAE  = mean(abs(y_train_predict(:) - train_y(:)));
train_MAPE = mean(abs((y_train_predict(:) - train_y(:)) ./ train_y(:)));
train_MSE  = mean((y_train_predict(:) - train_y(:)).^2);
train_RMSE = sqrt(train_MSE);
train_R2   = 1 - mean(norm(train_y - y_train_predict)^2 / norm(train_y - mean(train_y))^2);

disp([Tvalue, ' Train MAE:  ', num2str(train_MAE)]);
disp([Tvalue, ' Train MAPE: ', num2str(train_MAPE)]);
disp([Tvalue, ' Train MSE:  ', num2str(train_MSE)]);
disp([Tvalue, ' Train RMSE: ', num2str(train_RMSE)]);
disp([Tvalue, ' Train R^2:  ', num2str(train_R2)]);
disp('************************************************************************************');

vaild_y = vaild_y_feature_label;
vaild_MAE  = mean(abs(y_vaild_predict(:) - vaild_y(:)));
vaild_MAPE = mean(abs((y_vaild_predict(:) - vaild_y(:)) ./ vaild_y(:)));
vaild_MSE  = mean((y_vaild_predict(:) - vaild_y(:)).^2);
vaild_RMSE = sqrt(vaild_MSE);
vaild_R2   = 1 - mean(norm(vaild_y - y_vaild_predict)^2 / norm(vaild_y - mean(vaild_y))^2);

disp([Tvalue, ' Valid MAE:  ', num2str(vaild_MAE)]);
disp([Tvalue, ' Valid MAPE: ', num2str(vaild_MAPE)]);
disp([Tvalue, ' Valid MSE:  ', num2str(vaild_MSE)]);
disp([Tvalue, ' Valid RMSE: ', num2str(vaild_RMSE)]);
disp([Tvalue, ' Valid R^2:  ', num2str(vaild_R2)]);
disp('************************************************************************************');

test_y = test_y_feature_label;
test_MAE  = mean(abs(y_test_predict(:) - test_y(:)));
test_MAPE = mean(abs((y_test_predict(:) - test_y(:)) ./ test_y(:)));
test_MSE  = mean((y_test_predict(:) - test_y(:)).^2);
test_RMSE = sqrt(test_MSE);
test_R2   = 1 - mean(norm(test_y - y_test_predict)^2 / norm(test_y - mean(test_y))^2);

disp([Tvalue, ' Test MAE:  ', num2str(test_MAE)]);
disp([Tvalue, ' Test MAPE: ', num2str(test_MAPE)]);
disp([Tvalue, ' Test MSE:  ', num2str(test_MSE)]);
disp([Tvalue, ' Test RMSE: ', num2str(test_RMSE)]);
disp([Tvalue, ' Test R^2:  ', num2str(test_R2)]);

%% ==========================================================
%  Local functions
% ==========================================================
function [alpha_opt, best_rmse] = optimize_vmd_alpha_pso(signal, K, lb, ub, popSize, maxIters)
% Optimize VMD penalty parameter alpha by population-based metaheuristic (PSO)
% Objective: minimize reconstruction RMSE.

    % If Global Optimization Toolbox is available, use particleswarm
    if exist('particleswarm', 'file') == 2
        objFun = @(a) vmd_recon_rmse(signal, K, a);
        opts = optimoptions('particleswarm', ...
            'SwarmSize', popSize, ...
            'MaxIterations', maxIters, ...
            'Display', 'iter', ...
            'UseParallel', false);
        [alpha_opt, best_rmse] = particleswarm(objFun, 1, lb, ub, opts);
        return;
    end

    % Otherwise, use a lightweight PSO implementation (no toolbox)
    objFun = @(a) vmd_recon_rmse(signal, K, a);

    % PSO hyperparameters (reasonable defaults)
    w  = 0.72;     % inertia
    c1 = 1.49;     % cognitive
    c2 = 1.49;     % social

    % Initialize swarm
    pos = lb + (ub - lb) * rand(popSize, 1);
    vel = zeros(popSize, 1);

    pbest_pos = pos;
    pbest_val = inf(popSize, 1);

    [gbest_val, gbest_idx] = min(pbest_val);
    gbest_pos = pbest_pos(gbest_idx);

    for it = 1:maxIters
        for i = 1:popSize
            val = objFun(pos(i));
            if val < pbest_val(i)
                pbest_val(i) = val;
                pbest_pos(i) = pos(i);
            end
        end

        [cur_best, idx] = min(pbest_val);
        if cur_best < gbest_val
            gbest_val = cur_best;
            gbest_pos = pbest_pos(idx);
        end

        % Update velocity & position
        r1 = rand(popSize,1);
        r2 = rand(popSize,1);
        vel = w*vel + c1*r1.*(pbest_pos - pos) + c2*r2.*(gbest_pos - pos);
        pos = pos + vel;

        % Clamp to bounds
        pos = max(min(pos, ub), lb);

        disp(['[PSO] iter = ', num2str(it), ...
              ', best_RMSE = ', num2str(gbest_val), ...
              ', best_alpha = ', num2str(gbest_pos)]);
    end

    alpha_opt = gbest_pos;
    best_rmse = gbest_val;
end

function rmse = vmd_recon_rmse(signal, K, alpha)
% VMD decomposition with specified alpha; evaluate reconstruction RMSE.
    [imf, res] = vmd(signal, 'NumIMF', K, 'Alpha', alpha);
    recon = sum(imf, 2) + res;
    err = recon - signal;
    rmse = sqrt(mean(err.^2));
end
