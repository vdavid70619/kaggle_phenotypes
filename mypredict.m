function predict_probs = mypredict(tests, model, dbc)
    addpath('../../LIB/liblinear-1.93/matlab');
    addpath('../../LIB/dbc');
    addpath('../../LIB/libsvm-3.18/matlab');

%     tests = normalize(tests);    
%     codes=DBC_apply(tests',model.dbc)';

    codes = tests;

%     codes = vl_hikmeanspush(model.tree,uint8(tests'))';

    %codes = compute_mapping(tests, 'PCA', 50);
    codes = sparse(double(codes));
    codes = mynormalize(codes);
    
    labels = zeros(size(codes,1),1);
    %[predict_labels, accs, prob] = predict(labels, codes, model, '-b 1');
    [predict_label, accs, prob] = svmpredict(labels, codes, model.svm, '-b 1');
    %prob = normalize(prob);  
    if isempty(prob)
        predict_probs = predict_label;        
    else
        predict_probs = prob;
    end
end