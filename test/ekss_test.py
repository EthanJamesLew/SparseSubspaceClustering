'''
Tests for ekss.py
'''

import unittest
from numpy.random import uniform
from numpy import allclose
from numpy import eye
import random
from ssc.ekss import unif_bases

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


if __name__=="__main__":
    unittest.main()
