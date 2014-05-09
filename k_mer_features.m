function hist = k_mer_features(data, K)
    hist = [];
    for j=1:length(K)
        k = K(j)
        onefilter = ones(k,1);
        k_mers = imfilter(data, onefilter, 'conv');
        k_mers = k_mers(:, ceil(k/2):end-ceil(k/2));

        k_ind = unique(k_mers);
        k_ind = sort(k_ind);

        hist1 = zeros(size(data,1),length(k_ind));
        for i=1:length(k_ind)
            hist1(:,i) = sum(k_mers==k_ind(i), 2);
        end
        hist1 = mynormalize(hist1);
        
        hist = [hist hist1];
    end
    hist;
end