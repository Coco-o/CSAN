import os
import numpy as np
from sklearn.decomposition import PCA

def load_data(filename, delim):
    data = []
    with open(filename) as infile:
        for line in infile:
            data.append(line.rstrip('\n').split(delim))
    return data

def process_data(path, filename):
    dataset = filename.rstrip('.data')
    if dataset == 'abalone':
        raw_data = load_data(os.path.join(path, filename), ',')
        data = np.array(raw_data)
        X = np.array(data[:,1:8], dtype=float)
        Y = np.array(data[:,8], dtype=int)
    return X, Y

# find the index of nearest sample in the data
def find_nearest(data, val):
    indx = (np.abs(data - val)).argmin()
    return indx

def subsample(X, Y, pca_X, mu, sigma, num_sample):
    sample_X = []
    sample_Y = []
    for i in range(num_sample):
        sample = np.random.normal(mu, sigma)
        indx = find_nearest(pca_X, sample)
        sample_X.append(X[indx, :])
        sample_Y.append(Y[indx])

        # remove data once sampled
        pca_X = np.delete(pca_X, indx, axis=0)
        X = np.delete(X, indx, axis=0)
        Y = np.delete(Y, indx, axis=0)

    return sample_X, sample_Y, pca_X, X, Y

def generate_real(path, filename):
    X, Y = process_data(path, filename)

    # pca & projection to first principle component
    pca = PCA(n_components=1)
    pca_X = pca.fit_transform(X)

    mu_train = np.min(pca_X)
    mu_test = np.max(pca_X)
    sigma = 0.5 * np.std(pca_X)

    # subsample data
    test_X, test_Y, pca_X, X, Y = subsample(X, Y, pca_X, mu_test, sigma, 200)
    train_X, train_Y, pca_X, X, Y = subsample(X, Y, pca_X, mu_train, sigma, 500)

    return train_X, train_Y, test_X, test_Y

def main():
    train_X, train_Y, test_X, test_Y = generate_real('../UCI', 'abalone.data')

if __name__=='__main__':
    main()
