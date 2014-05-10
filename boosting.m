function boosting()
    addpath(genpath('../../LIB/drtoolbox'));

    train_data = load('data_cache.mat');
    train.samples = cat(2, train_data.data.feature{:})';    
    train.labels = train_data.data.label;
    train.population = train_data.data.population;
    
    %% Shuffle data
    rand_indx = randperm(size(train.samples,1));    
    samples = train.samples(rand_indx,:);
%     k_mers = train.k_mers(rand_indx,:);
    labels = train.labels(rand_indx,:);
    populations = train.population(rand_indx);  
    
    load('dim_reduct.mat')
    useful_dims = useful_dims(1:1000);  
    samples = samples(:,useful_dims);     

%     samples = compute_mapping(samples, 'ProbPCA', 100);    
    train_samples = [samples  populations]; %k_mers
    train_labels = labels;
    
    X = train_samples;
    Y = train_labels;
 
    %% Cross validation
    cvpart = cvpartition(Y,'k',10);    
    for cvi = 1:cvpart.NumTestSets

        Xtrain = X(cvpart.training(cvi),:);
        Ytrain = Y(cvpart.training(cvi),:);
        Xtest = X(cvpart.test(cvi),:);
        Ytest = Y(cvpart.test(cvi),:); 

        fun = @(Y,Yfit,W) logloss(Y,Yfit);

        boost{cvi} = fitensemble(Xtrain,Ytrain,'Bag',100,'Tree','type','regression');
        lloss{cvi} = loss(boost{cvi}, Xtest, Ytest, 'lossfun', fun)
        Yfit = predict(boost{cvi}, Xtest);
        fprintf('Log loss: %f\n', logloss(Ytest, Yfit))
    end
    
    losses = cat(1,lloss{:});
    s_losses = sort(cat(1,lloss{:}));
    s_id = find(s_losses >= median(s_losses),1);
    m_ind = find(losses == s_losses(s_id));
    fprintf('Mean Logloss: %f\n', mean(losses));
    fprintf('VAR Logloss: %f\n', var(cat(1,lloss{:})));
    fprintf('Best Logloss: %f, Model: %d\n', lloss{m_ind}, m_ind);
    best_model = boost{m_ind};
    
    model = best_model;
    save('model.mat','model');
    
    %% Generate kaggle result
    test_data = load('test_cache.mat');    
    
    test.samples = cat(2, test_data.data.feature{:})';    
    test.populations = test_data.data.population;
    test.ids = test_data.data.id;      
    
    test.samples = test.samples(:,useful_dims); 
%     test.samples = compute_mapping(test.samples, 'ProbPCA', 100);    
    
    
    kaggle_test.samples = [test.samples  test.populations]; %test.k_mers
    kaggle_test.id = test.ids;
    generate_kaggle_boost_result(kaggle_test, best_model);    
end