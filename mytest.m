function [accs, loss] = mytest(tests, labels, model)
    fprintf('>>>>>>>>>>>>>>>>>>Testing\n');
    addpath('../../LIB/liblinear-1.93/matlab');
    addpath('../../LIB/dbc');
    addpath('../../LIB/libsvm-3.18/matlab');

%     tests = normalize(tests);    
%     codes=DBC_apply(tests',model.dbc)';

    codes = tests;
%     codes = compute_mapping(tests, 'PCA', 100);

    codes = vl_homkermap(double(codes'), 1, 'kchi2', 'gamma', .5)';
    codes = codes*10;

    codes = vl_hikmeanspush(model.tree,uint8(codes'))';
%     codes = vl_homkermap(double(codes'), 1, 'kchi2', 'gamma', .5)';
    
    codes = sparse(double(codes));
    codes = mynormalize(codes);
    
    [predict_labels, accs, prob] = predict(labels, codes, model.svm, '-b 1');
    %[predict_label, accs, prob] = svmpredict(labels, codes, model.svm, '-b 1');
    %disp(prob)    
    if isempty(prob)
        loss = logloss(labels, predict_label);        
    else
        loss = logloss(labels, prob(:,1));
    end
    fprintf('LogLoss: %f\n',loss);
end