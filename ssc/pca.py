from numpy import argsort, array, cov, dot, newaxis
from scipy.linalg import eigh

def pca(data, r):
    '''
    input: data (D dimensions x N observations), r rank of PCA
    output: PCA
    '''

    if data.ndim < 2:
        data = array(data, ndmin=2, copy=False)
        data = data.T
    if data.size == 0:
        return data
    data -= data.mean(axis=0)
    R = cov(data, rowvar=False)
    R = array(R, ndmin=2, copy=False)

    evals, evecs = eigh(R)
    idx = argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    evecs = evecs[:, :r]
    return (evecs.T @ data.T).T
