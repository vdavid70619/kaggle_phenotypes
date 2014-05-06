'''
Have to use python for this. It is easy.
Xiyang
'''

import scipy.io
import scipy as sp
import numpy as np

from sklearn import svm
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import cluster
from sklearn import neighbors
from sklearn import tree
from sklearn import naive_bayes

def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def main():
    train = scipy.io.loadmat('data.mat');
    print np.shape(train['data'])
    print np.shape(train['label'])

    X = preprocessing.scale(train['data'])
    y = train['label']
    y = y.transpose()[0]

    train_data = X[100:700]
    test_data = np.concatenate((X[:100], X[700:800]))
    train_label = y[100:700]
    test_label = np.concatenate((y[:100], y[700:800]))

    clustering = cluster.SpectralClustering(n_clusters=100)
    clustering.fit(train_data)
    train_data = clustering.transform(train_data)

    print np.shape(train_data)

    test_data = clustering.transform(test_data)

    #decomp = decomposition.MiniBatchSparsePCA(n_components=100, verbose=True)
    #train_data = decomp.fit_transform(train_data)
    #test_data = decomp.fit_transform(test_data)

    #regressor = linear_model.ARDRegression()
    #regressor = naive_bayes.GaussianNB()
    #regressor = tree.ExtraTreeRegressor()
    #regressor = neighbors.KNeighborsRegressor(n_neighbors=100, algorithm='auto')
    #regressor = neighbors.RadiusNeighborsRegressor(radius=1.0)
    #regressor = linear_model.Lasso()
    regressor = svm.SVR(verbose=True, cache_size=4000)
    regressor.fit(train_data, train_label)
    test_predict = regressor.predict(test_data)

    print test_predict

    print llfun(test_label, test_predict)


if __name__ == "__main__":
    main()


