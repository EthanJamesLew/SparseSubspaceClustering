'''
Ensemble K-Subspaces
John Lipor, David Hong, Dejiao Zhang, and Laura Balzano
Author: Ethan Lew
5/23/18
'''

from numpy.linalg import norm
from scipy.linalg import orth, svd
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
from numpy.random import uniform
from numpy import argmin, argsort, array, diag, dot, squeeze, where, zeros
from pca import pca

def ekss (X, k, r, maxIter):
    '''
    Ensemble K-Subspaces
    Input: X data (D dim x N observations), k clusters, r subspace rank, maxIter
    iterations
    Output: Y (N labels of K subspaces)
    '''
    #TODO: Make this into an interator?
    # Get dimensions
    D, N = X.shape;
    # Draw K random subspace bases
    U = unif_bases(D, k, r);
    # Cluster by nearest subspace
    C = cluster_proj(U, X, k);
    # Estimate subspaces
    for ii in range(0, maxIter):
        U = pca_bases(C, k, r);
        C = cluster_proj(U, X, k);
    # Cluster by nearest subspace

def unif_bases(D, k, r):
    '''
    Create K Uniform Subspace basis
    Input: D dimension of data, r candidate dimension, K bases (subspaces)
    Output: U interator with uniformly random bases on unit sphere
    '''
    for ii in range(0, k):
        yield orth(uniform(0, 1, (D, r)));


def pca_bases(C, k, d):
    '''
    Estimate Subspaces with PCA
    Inputs: C cluster (dictionary with keys 1-6), k clusters and d candidate dim
    Outputs: U iterator with bases estimated via PCA
    '''
    for ii in range(0, k):
        yield pca(X, k)


def cluster_proj(U, X, k):
    '''
    Cluster by Projection
    Input: U iterator of subspaces, X data, k clusters
    Output: C dictionary, where c_k is a set of vectors satisfying the projection
    '''
    D,N  = X.shape;
    X_arg = zeros((k, N));
    ii = 0;
    for uk in U:
        X_arg[ii, :] = norm(uk.T @ X, 2, 0).T;
        ii = ii + 1;
    X_ind = argsort(X_arg, 0);
    X_max = X_ind[k-1, :];
    C = {};
    for ii in range(0, k):
        ind = where(X_max == ii);
        L = X[ :, ind];
        C[ii] = squeeze(L);
    return C


def cluster_nss(U, X, k):
    '''
    Clusters x to nearest subspace (nss)
    Input: U iterator of subspaces, X data, k clusters
    Output: k label, ranging from 0-(K-1)
    '''
    D,N  = X.shape;
    X_arg = zeros((k, N));
    ii = 0;
    for uk in U:
        uk_cov = uk @ uk.T
        X_arg[ii, :] = norm(X - uk_cov @ X, 2, 0).T;
        ii = ii + 1;
    return argmin(X_arg, axis=0)


if __name__ == "__main__":
    X= uniform(20, -20, (5, 10));
    ekss(X, 3, 2, 100);
