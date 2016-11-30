import nose
import numpy as np
from mat_mul import *

# setup
mul_funcs = [naive_mat_mul, recursive_mat_mul, strassen_mat_mul]
qr_funcs = [givens_QR, house_QR]
eig_funcs = [eig, hessenberg_eig]
n = [2**x for x in range(4)]


# mat_mul tests

def test_real():
    # test mat_mul for real valued matrices
    for m in n:
        A = np.random.random((m, m))
        B = np.random.random((m, m))
        for func in mul_funcs:
            assert np.allclose(func(A, B), np.dot(A, B))


def test_int():
    # test mat_mul for natural valued matrices
    for m in n:
        A = np.random.randint(0, 100, (m, m))
        B = np.random.randint(0, 100, (m, m))
        for func in mul_funcs:
            assert np.allclose(func(A, B), np.dot(A, B))


def test_neg():
    # test mat_mul for integer valued matrices
    for m in n:
        A = np.random.randint(-100, 100, (m, m))
        B = np.random.randint(-100, 100, (m, m))
        for func in mul_funcs:
            assert np.allclose(func(A, B), np.dot(A, B))


def test_complex():
    # test mat_mul for complex valued matrices
    for m in n:
        A = np.random.randint(-10, 10, (m, m)) + 1j*np.random.randint(0, 10, (m, m))
        B = np.random.randint(-10, 10, (m, m)) + 1j*np.random.randint(0, 10, (m, m))
        for func in mul_funcs:
            assert np.allclose(func(A, B), np.dot(A, B))


# QR/eigenvalue tests

def test_QR():
    # test QR decompositions
    for m in n:
        for func in qr_funcs:
            A = np.random.randint(-10, 10, (m, m))
            Q, R = func(A)
            # show Q*R == A
            assert np.allclose(Q.dot(R), A)
            # show that Q is orthonormal
            assert np.allclose(Q.dot(Q.T), np.identity(m))
            # show that R is upper triangular
            assert np.allclose(R, np.triu(R))



def test_hessenberg():
    # tests for hessenberg functions
    for m in n:
        A = np.random.randint(-10, 10, (m, m))

        # test for hessenberg reduction
        Q, H = hessenberg(A)
        # show that all entries below the first lower subdiagonal are 0
        for i in range(m-2):
            assert np.allclose(0, H[i+2:, i])
        # show that Q is orthonormal
        assert np.allclose(Q.dot(Q.T), np.identity(m))
        # show that Q*H*Q.T == A
        assert np.allclose(Q.dot(H).dot(Q.T), A)

        # test for QR decomposition of an upper hessenberg matrix
        R, Gs = hessenberg_QR(H)
        # show that R is upper triangular hessenberg
        assert np.allclose(R, np.triu(R))
        #for i in range(m-2):
        #    assert np.allclose(0, R[i+2:, i])
        # show that the for Q = G[n-1]*G[n-2]*...*G[0], Q*R == H
        for g in Gs[::-1]:
            R = np.dot(g, R)
        assert np.allclose(R, H)


def test_eig():
    for m in n:
        A = np.random.randint(-10, 10, (m, m))
        # test is only for symetric matrices, which have real eigenvalues
        A = A.dot(A.T)
        eig_vals = np.linalg.eig(A)[0]
        for func in eig_funcs:
            assert np.allclose(func(A, iterations = 100), np.sort(eig_vals))
