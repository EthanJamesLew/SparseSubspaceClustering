from numpy import argsort, cov, dot
from scipy.linalg import eigh

def pca(data, r):
    '''
    input: data (D dimensions x N observations), r rank of PCA
    output: PCA
    '''
    m, n = data.shape
    data -= data.mean(axis=0)
    R = cov(data, rowvar=False)
    evals, evecs = eigh(R)
    idx = argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    evecs = evecs[:, :r]
    return dot(evecs.T, data.T).T
