function model = mynntrain(trains, labels)
    fprintf('>>>>>>>>>>>>>>>>>>Training\n');
    addpath(genpath('../../LIB/DeepLearnToolbox'));    
    % normalize
    [train_x, mu, sigma] = zscore(trains);   
    train_y = [1-labels labels];
    %train_y = labels;
    nn = nnsetup([897 2]);
    
    nn.nonSparsePenalty      = 10;
    nn.weightPenaltyL2      = 1e-2;         %  L2 weight decay
    nn.dropoutFraction      = 0.5;          %  Dropout fraction
    nn.activation_function  = 'sigm';       %  Sigmoid activation function
    nn.learningRate         = 1;            %  Sigm require a lower learning rate    
    nn.output               = 'softmax';    %  use softmax output
    opts.numepochs          = 100;            %  Number of full sweeps through data
    opts.batchsize          = 10;           %  Take a mean gradient step over this many samples
    opts.plot               = 0;          	%  enable plotting    
    
    nn = nntrain(nn, train_x, train_y, opts);
    [er, bad] = nntest(nn, train_x, train_y);
    [~, regress]= nnpredict(nn, train_x);
    loss = logloss(labels, regress(:,end));        
    confusion_matrix(labels, regress(:,end)>0.5);
    fprintf('LogLoss: %f\n',loss);    
    
    model.nn = nn;
    model.mu = mu;
    model.sigma = sigma;
end