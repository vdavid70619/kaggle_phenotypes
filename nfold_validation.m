%%
%%
%%
function best_model = nfold_validation()
    N = 8
    
    %% Load train data
    train_data = load('data_cache.mat');
    train.samples = cat(2, train_data.data.feature{:})';    
    train.labels = train_data.data.label;
    train.population = train_data.data.population;
    
    %% Load test data
    test_data = load('test_cache.mat');  
    test.samples = cat(2, test_data.data.feature{:})'; 
    test.populations = test_data.data.population;
    test.ids = test_data.data.id;
%     
%     %% Get k_mers
%     k_mers = k_mer_features([train.samples; test.samples], [4 8 16 32 64 128]);
%     test.k_mers = k_mers(801:1000,:);
%     train.k_mers = k_mers(1:800,:);
%     
    %% Do dim reduction on samples
    useful_dims = [];    
    load('dim_reduct.mat')
    train.samples = train.samples(:,useful_dims); 
    test.samples = test.samples(:,useful_dims);   

    %% Shuffle
    rand_indx = randperm(size(train.samples,1));
    samples = train.samples(rand_indx,:);
%     k_mers = train.k_mers(rand_indx,:);
    labels = train.labels(rand_indx,:);
    populations = train.population(rand_indx);
    
%     %% GMM encoding
%     load('gmm_model.mat');
%     ncodes = 5;
%     encode = zeros(size(samples,1),ncodes);
%     for i=1:size(samples,1)
%         if mod(i,100)==0 
%             fprintf('.');
%         end
%         enc = vl_fisher(samples(i,:)', gmms.means, gmms.covariances, gmms.priors, 'Fast')';
%         dsize = length(enc);
%         dstep = floor(dsize/ncodes);
%         for j=1:ncodes
%             encode(i,j) = sum(enc((j-1)*dstep+1:j*dstep));
%         end
%     end
%     samples = encode;
   
    samples = [samples populations];

    cvpart = cvpartition(labels,'k',10);    
    for cvi = 1:cvpart.NumTestSets

        Xtrain = samples(cvpart.training(cvi),:);
        Ytrain = labels(cvpart.training(cvi),:);
        Xtest = samples(cvpart.test(cvi),:);
        Ytest = labels(cvpart.test(cvi),:);   
        
        model{cvi} = mytrain(Xtrain,Ytrain);
        %model{i} = mynntrain(train_samples,train_labels);
        [acc, loss{cvi}] = mytest(Xtest,Ytest, model{cvi});
        %[acc, loss{i}] = mynntest(test_samples,test_labels, model{i});
        accuracy{cvi} = acc(1);
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
    save('model.mat','model');
    
    
    tests.samples = [test.samples  test.populations]; %test.k_mers
    tests.id = test.ids;
    generate_kaggle_result(tests, model);
end
