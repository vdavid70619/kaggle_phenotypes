function best_model = nfold_validation()
    N = 4
    
    load('data_cache.mat');
    
    samples = cat(2, data.feature{:})';    
    labels = data.label;
    
    load('dim_reduct.mat');
    
    samples = samples(:, useful_dims);
    samples = samples(randperm(size(samples,1)),:);
        
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
                
        %model{i} = mytrain(train_samples,train_labels);
        model{i} = mynntrain(train_samples,train_labels);
        %[acc, loss{i}] = mytest(test_samples,test_labels, model{i});
        [acc, loss{i}] = mynntest(test_samples,test_labels, model{i});
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
    save('model.mat','model','useful_dims');
end
