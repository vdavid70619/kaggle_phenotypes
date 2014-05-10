function generate_kaggle_result(tests, model)
    
%     MODELFILE = 'model.mat'
%     load(MODELFILE);

    SAVEFILE = 'kaggle.csv'
 
%     if ~exist('test_cache.mat')
%         data = dataloader('-df', 'data/test_genotypes.csv', '-cf', 'test_cache.mat');
%     else
%         load('test_cache.mat')
%     end

    samples = tests.samples; 
    ids = tests.id;
    
    
    fd = fopen(SAVEFILE, 'w+');
    fprintf(fd,'Id,Category\n');
    
    
    %samples = samples(:, useful_dims(1:1000));    
    %predict_probs = mynnpredict(samples, model);
    predict_probs = mypredict(samples, model);

    for i=1:size(predict_probs,1)
        fprintf(fd,'%s,%f\n', ids{i}, predict_probs(i));       
    end  
    fclose(fd);
 
end