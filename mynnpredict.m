function predict_probs = mynnpredict(tests, model)
    addpath(genpath('../../LIB/DeepLearnToolbox'));

    % normalize
    test_x = normalize(tests, model.mu, model.sigma);
    
    %[er, bad] = nntest(model.nn, test_x, test_y);
    [~, regress]= nnpredict(model.nn, test_x);
    predict_probs = regress;
end