function generate_kaggle_result()
    
    MODELFILE = 'model.mat'
    SAVEFILE = 'kaggle.txt'
 
    if ~exist('test_cache.mat')
        data = dataloader('-df', 'data/test_genotypes.csv', '-cf', 'test_cache.mat');
    else
        load('test_cache.mat')
    end
    samples = cat(2, data.feature{:})'; 
    ids = data.id;
    
    
    fd = fopen(SAVEFILE, 'w+');
    fprintf(fd,'Id,Category\n');
    
    load(MODELFILE);
    
    predict_probs = mypredict(samples, model);
    
    for i=1:size(predict_probs,1)
        fprintf(fd,'%s,%f\n', ids{i}, predict_probs(i));       
    end  
    fclose(fd);
 
end