'''
Tests for ekss.py
'''

import unittest
from numpy.random import uniform, rand
from numpy.linalg import matrix_rank
from numpy import allclose, array, eye, zeros
import random
from ssc.ekss import unif_bases, pca_bases, cluster_proj
from ssc.pca import pca

class TestEkss(unittest.TestCase):
    def test_unif_bases(self):
        for ii in range(1, 4):
            # Construct a feasible array
            D = random.randrange(30, 1000);
            k = random.randrange(3, 20);
            r = random.randrange(2, D-1);
            U= unif_bases(D, k, r);
            for uk in U:
                # Check array shape
                m,n = uk.shape;
                self.assertEqual(m, D);
                self.assertEqual(n, r);
                # Check column vector orthogonality
                uk_gram = uk.T @ uk;
                allclose(uk_gram, eye(n) );
    def test_pca_bases(self):
        for ii in range(0, 5):
            D = random.randrange(5, 30)
            r = random.randrange(2, D-1)
            C = {0:rand(D, 1), 1:rand(D, 2), 2:rand(D, 3)}
            Cpca = pca_bases(C, 3, r)
            # Test dimensions of projection and rank (LR factorization)
            for ck in Cpca:
                m,n = ck.shape
                self.assertEqual(m, D)
                self.assertTrue(matrix_rank(ck) <= r)

    def test_cluster_proj(self):
        U = generate_basis_iterator()
        X = eye(3)
        # Add perturbations
        X = X + rand(3,3)/4
        #print(X)
        k = 3
        print(cluster_proj(U, X, k))

def generate_basis_iterator():
    yield array([[1],[0],[0]])
    yield array([[0],[1],[0]])
    yield array([[0],[0],[1]])
if __name__=="__main__":
    unittest.main()
