function dimension_reduct()
    rmpath(genpath('../../LIB/DeepLearnToolbox'));    
    train = load('data_cache.mat');
    test = load('test_cache.mat');

    train_samples = cat(2, train.data.feature{:})'; 
    train_labels = train.data.label;
 
    test_samples = cat(2, test.data.feature{:})';     
    n=20
    repeat_test_labels=[];
    repeat_test_samples=[];
    for i=1:n
        repeat_test_labels = [repeat_test_labels; rand(size(test_samples,1),1)];
        repeat_test_samples = [repeat_test_samples; test_samples];
    end
    repeat_test_labels(repeat_test_labels>0.5) = 1;
    repeat_test_labels(repeat_test_labels<=0.5) = 0;
    fprintf('Summary for generate random labels: %d/%d', sum(repeat_test_labels), length(repeat_test_labels));
    
    samples = [train_samples; repeat_test_samples];
    labels = [train_labels; repeat_test_labels];
    
    lscores = lasso(samples, labels);
    [~, ind] = sort(sum(abs(lscores),2), 'descend');
    useful_dims = ind(1:1000);
    save('dim_reduct.mat', 'useful_dims');
end