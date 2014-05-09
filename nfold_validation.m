%%
%% <TODO> K-mer features
%%
function best_model = nfold_validation()
    N = 8
    
    train_data = load('data_cache.mat');
    test_data = load('test_cache.mat');
    
    train_samples = cat(2, train_data.data.feature{:})';    
    train_labels = train_data.data.label;
  
    test_samples = cat(2, test_data.data.feature{:})'; 
%     load('dim_reduct.mat');
%     
%     samples = samples(:, useful_dims);
%     samples = samples(randperm(size(samples,1)),:);
   
    samples = k_mer_features([train_samples; test_samples], [4 8 16 32 64 128]);
    samples = samples(1:800,:);
    labels = train_labels;
    samples = samples(randperm(size(samples,1)),:);
    
    useful_dims = [];
    for i=1:N
        fprintf('============= Fold %d =============\n', i);
                
        pos_sample = samples(labels==1,:);
        neg_sample = samples(labels==0,:);
        
        pos_bin = round(size(pos_sample,1)/N);
        neg_bin = round(size(neg_sample,1)/N);
        
        test_pos_sample = pos_sample((i-1)*pos_bin+1:i*pos_bin,:);
        test_neg_sample = neg_sample((i-1)*neg_bin+1:i*neg_bin,:);
        
        pos_sample((i-1)*pos_bin+1:i*pos_bin,:) = [];
        neg_sample((i-1)*neg_bin+1:i*neg_bin,:) = [];
        
        test_samples = [test_pos_sample; test_neg_sample];
        test_labels = [ones(size(test_pos_sample,1),1); zeros(size(test_neg_sample,1),1)];

        train_samples = [pos_sample; neg_sample];
        train_labels = [ones(size(pos_sample,1),1); zeros(size(neg_sample,1),1)];    
                
        model{i} = mytrain(train_samples,train_labels);
        %model{i} = mynntrain(train_samples,train_labels);
        [acc, loss{i}] = mytest(test_samples,test_labels, model{i});
        %[acc, loss{i}] = mynntest(test_samples,test_labels, model{i});
        accuracy{i} = acc(1);
    end
    
    fprintf('Summary \n', mean(cat(1,accuracy{:})));    
    fprintf('Mean AC: %f \n', mean(cat(1,accuracy{:})));
    losses = cat(1,loss{:});
    fprintf('Logloss: %f \n', losses);
    fprintf('Mean Logloss: %f\n', mean(losses));
    s_losses = sort(cat(1,loss{:}));
    s_id = find(s_losses >= median(s_losses),1);
    m_ind = find(losses == s_losses(s_id));
    fprintf('VAR Logloss: %f\n', var(cat(1,loss{:})));
    fprintf('Median Logloss: %f, Model: %d\n', s_losses(s_id), m_ind);
    best_model = model{m_ind};
    model = best_model;
    model.useful_dims = useful_dims;
    save('model.mat','model');
end
