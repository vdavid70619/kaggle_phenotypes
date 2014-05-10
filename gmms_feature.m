function gmms_feature(numClusters)

    train_data = load('data_cache.mat');
    
    % data is (ndims,nsamples) for vl_fleat
    data = cat(2, train_data.data.feature{:});
    dimension = size(data,1);
    
    [initMeans, assignments] = vl_kmeans(data, numClusters, ...
        'Algorithm','Lloyd', ...
        'MaxNumIterations',5);

    initCovariances = zeros(dimension,numClusters);
    initPriors = zeros(1,numClusters);

    % Find the initial means, covariances and priors
    for i=1:numClusters
        data_k = data(:,assignments==i);
        initPriors(i) = size(data_k,2) / numClusters;

        if size(data_k,1) == 0 || size(data_k,2) == 0
            initCovariances(:,i) = diag(cov(data'));
        else
            initCovariances(:,i) = diag(cov(data_k'));
        end
    end

    % Run EM starting from the given parameters
    [means,covariances,priors,ll,posteriors] = vl_gmm(data, numClusters, ...
        'initialization','custom', ...
        'InitMeans',initMeans, ...
        'InitCovariances',initCovariances, ...
        'InitPriors',initPriors);
    
    gmms.means = means;
    gmms.covariances = covariances;
    gmms.priors = priors;
    save('gmm_model.mat', 'gmms');
end