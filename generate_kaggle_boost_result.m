function generate_kaggle_boost_result(tests, model)

    SAVEFILE = 'kaggle.csv'
    fd = fopen(SAVEFILE, 'w+');
    fprintf(fd,'Id,Category\n');
    
    samples = tests.samples; 
    ids = tests.id;
           
    predict_probs = predict(model, samples);

    for i=1:size(predict_probs,1)
        fprintf(fd,'%s,%f\n', ids{i}, predict_probs(i));       
    end  
    fclose('all');
 
end