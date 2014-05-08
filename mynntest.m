function [acc, loss] = mynntest(tests, labels, model)
    fprintf('>>>>>>>>>>>>>>>>>>Testing\n');
    addpath(genpath('../../LIB/DeepLearnToolbox'));

%     tests = tests(:, model.useful_dims);
    % normalize
    test_x = normalize(tests, model.mu, model.sigma);
    %test_y = [1-labels labels];
    test_y = labels;
    
    [er, bad] = nntest(model.nn, test_x, test_y);
    [~, regress]= nnpredict(model.nn, test_x);
    loss = logloss(labels, regress);        
    acc = confusion_matrix(labels, regress>0.5);
    fprintf('LogLoss: %f\n',loss);        
end