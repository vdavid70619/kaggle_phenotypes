function dimension_reduct()
    rmpath(genpath('../../LIB/DeepLearnToolbox'));    
    train = load('data_cache.mat');
    test = load('test_cache.mat');

    train_samples = cat(2, train.data.feature{:})'; 
    test_samples = cat(2, test.data.feature{:})'; 

    train_labels = train.data.label;
    test_labels = ones(size(test_samples,1),1).*0.5;
    
    samples = [train_samples; test_samples];
    labels = [train_labels; test_labels];
    
    lscores = lasso(samples, labels);
    [~, ind] = sort(sum(abs(lscores),2), 'descend');
    useful_dims = ind(1:1000);
    save('dim_reduct.mat', 'useful_dims');
end