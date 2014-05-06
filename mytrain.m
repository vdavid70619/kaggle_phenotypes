function [model] = mytrain(trains, labels)
    fprintf('>>>>>>>>>>>>>>>>>>Training\n');
    run('../../LIB/vlfeat-0.9.17/toolbox/vl_setup.m');
    addpath('../../LIB/liblinear-1.93/matlab');
    addpath('../../LIB/dbc');
    addpath(genpath('../../LIB/drtoolbox'));
    addpath('../../LIB/libsvm-3.18/matlab');
    
%     trains = normalize(trains);
%     model.dbc=DBC_train(trains',labels', 16, '-B 1 -c 1 -s 1'); 
%     codes=DBC_apply(trains',model.dbc)';

    
    K =3
    nleaves = 1000
    [tree,A] = vl_hikmeans(uint8(trains'),K,nleaves, 'Verbose');  
    model.tree = tree;
    codes = vl_hikmeanspush(model.tree,uint8(trains'))';

    %trains = normalize(trains);
    %codes = compute_mapping(trains, 'LDA', 50);
    codes = sparse(double(codes));
    codes = normalize(codes);

    %model = train(labels, codes, '-s 0 -B 1 -c 10');
    model.svm = svmtrain(labels, codes, ['-s 4 -b 1 -t 3 -m 8000']);
        
    %[predict_labels, accs, prob] = predict(labels, codes, model, '-b 1');
    [predict_label, accs, prob] = svmpredict(labels, codes, model.svm, '-b 1');
    %prob = normalize(prob);
    if isempty(prob)
        loss = logloss(labels, predict_label);        
    else
        loss = logloss(labels, prob(:,1));
    end
    fprintf('LogLoss: %f\n',loss);
end