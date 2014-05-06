function [accs, loss] = mytest(tests, labels, model)
    fprintf('>>>>>>>>>>>>>>>>>>Testing\n');
    addpath('../../LIB/liblinear-1.93/matlab');
    addpath('../../LIB/dbc');
    addpath('../../LIB/libsvm-3.18/matlab');

%     tests = normalize(tests);    
%     codes=DBC_apply(tests',model.dbc)';



    codes = vl_hikmeanspush(model.tree,uint8(tests'))';

    %codes = compute_mapping(tests, 'PCA', 50);
    codes = sparse(double(codes));
    codes = normalize(codes);
    
    %[predict_labels, accs, prob] = predict(labels, codes, model, '-b 1');
    [predict_label, accs, prob] = svmpredict(labels, codes, model.svm, '-b 1');
    if isempty(prob)
        loss = logloss(labels, predict_label);        
    else
        loss = logloss(labels, prob(:,1));
    end
    fprintf('LogLoss: %f\n',loss);
end