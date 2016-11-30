'''created 9/25/16 by Charles Franzen

Notes
-----
This module implements:

* three matrix multiplication algorithms:
    -naive iterative
    -naive recursive
    -Strassen recursive

* two QR decompositions:
    -Givens rotations
    -Householder reflections

* two Hessenberg algorithms:
    -upper Hessenberg reduction
    -QR decomp of upper Hessenberg matrices

* the QR algorithm for real eigenvalues, both with and without Hessenberg
  reduction.

This was created as part of a final project for DSCI6001 at GalvanizeU.
'''

import numpy as np
import numpy.linalg as LA
import time


# matrix multiplication

def naive_mat_mul(A, B):
    '''
    Naive iterative method of matrix multiplication

    Inputs- A, B (np.array): two matrices of the same dimensions 2^n x 2^n

    Output- (np.array): a matrix of dimensions 2^n x 2^n
    '''
    # get matrix dimensions
    n = A.shape[0]
    # initialize result matrix to all zeros
    C = np.zeros((n, n), dtype=np.complex_)
    # iterate through rows and columns to build up dot products
    for i in range(n):
        for l in range(n):
            for k in range(n):
                C[i, l] += A[i, k] * B[k, l]
    return C


def recursive_mat_mul(A, B):
    '''
    Naive recursive method of matrix multiplication

    Inputs- A, B (np.array): two matrices of the same dimensions 2^n x 2^n

    Output- (np.array): a matrix of dimensions 2^n x 2^n
    '''
    # get matrix dimensions
    n = A.shape[0]
    # base case: 1x1 matrices
    if n == 1:
        return A * B
    # recursive case
    else:
        # get dimensions of sub-matrices
        i = int(n / 2)
        # initialize result matrix
        C = np.zeros((n, n), dtype=np.complex_)

        # calculate each submatrix as an addition of two matrix products
        # note that this requires 8 matrix multiplications
        C[:i, :i] = recursive_mat_mul(A[:i, :i], B[:i, :i])\
            + recursive_mat_mul(A[:i, i:], B[i:, :i])

        C[:i, i:] = recursive_mat_mul(A[:i, :i], B[:i, i:])\
            + recursive_mat_mul(A[:i, i:], B[i:, i:])

        C[i:, :i] = recursive_mat_mul(A[i:, :i], B[:i, :i])\
            + recursive_mat_mul(A[i:, i:], B[i:, :i])

        C[i:, i:] = recursive_mat_mul(A[i:, :i], B[:i, i:])\
            + recursive_mat_mul(A[i:, i:], B[i:, i:])

        return C


def strassen_mat_mul(A, B):
    '''
    Strassen's recursive method of matrix multiplication

    Inputs- A, B (np.array): two matrices of the same dimensions 2^n x 2^n

    Output- (np.array): a matrix of dimensions 2^n x 2^n
    '''
    # get matrix dimensions
    n = A.shape[0]
    # base case: 1x1 matrices
    if n == 1:
        return A * B
    # recursive case
    else:
        # get dimensions of sub-matrices
        i = int(n / 2)
        # initialize result matrix
        C = np.zeros((n, n), dtype=np.complex_)

        # calculate intermediate matrices.
        # note that this requires only 7 matrix multiplications
        M1 = strassen_mat_mul(A[:i, :i] + A[i:, i:], B[:i, :i] + B[i:, i:])
        M2 = strassen_mat_mul(A[i:, :i] + A[i:, i:], B[:i, :i])
        M3 = strassen_mat_mul(A[:i, :i], B[:i, i:] - B[i:, i:])
        M4 = strassen_mat_mul(A[i:, i:], B[i:, :i] - B[:i, :i])
        M5 = strassen_mat_mul(A[:i, :i] + A[:i, i:], B[i:, i:])
        M6 = strassen_mat_mul(A[i:, :i] - A[:i, :i], B[:i, :i] + B[:i, i:])
        M7 = strassen_mat_mul(A[:i, i:] - A[i:, i:], B[i:, :i] + B[i:, i:])

        # calculate sub-matrices by addition of the intermediate matrices
        C[:i, :i] = M1 + M4 - M5 + M7
        C[:i, i:] = M3 + M5
        C[i:, :i] = M2 + M4
        C[i:, i:] = M1 - M2 + M3 + M6

        return C


def mat_time(n, func, number=1000, repeats=3):
    '''
    Records running times for matrix multiplications

    Inputs- n (int): size of test matrices. must be a power of 2
            func (function): matrix multiplication function to be used
            num_trials (int): number of trials for the experiment
            repeats (int): number of experiments

    Output- list: times for each trial
    '''
    # create random n x n matrices
    A = np.random.randint(0, n**2, (n, n))
    B = np.random.randint(0, n**2, (n, n))

    def _timer(A, B):
        t1 = time.time()
        func(A, B)
        t2 = time.time()
        return t2 - t1
    # run all trials
    trials = [np.mean([_timer(A, B) for _ in range(number)])
              for _ in range(repeats)]
    return trials


# QR decomp and the QR algorithm

def _givens(a, b):
    '''
    Subroutine to calculate a rotation matrix g: [gamma, -sigma]
                                                 [sigma, gamma]
    such that g*[a] = [r]
                [b]   [0]

    This is a linear transformation that zeros the lower left element of
    a matrix M

    Inputs- a, b (float): values from the matrix to be decomposed

    Outputs- gamma, sigma (float): scalars used to construct a givens rotation
                                   matrix
    '''
    if np.abs(b) >= np.abs(a):
        tau = a / b
        sigma = 1 / np.sqrt(1 + tau**2)
        gamma = sigma * tau
    else:
        tau = b / a
        gamma = 1 / np.sqrt(1 + tau**2)
        sigma = gamma * tau
    return gamma, sigma


def givens_QR(A):
    '''
    Uses givens rotations to perform QR decomposition of A

    Input- A (np.array): the matrix to be decomposed

    Outputs- Q, R (np.array): an orthonormal matrix Q and an upper triangular
                              matrix R such that Q*R=A
    '''
    # get dimensions of a
    n, m = A.shape
    # initialize Q
    Q = np.identity(n)
    # initialize R
    R = np.copy(A)
    # for n columns
    for j in range(n):
        # for rows j-m
        for i in range(m - 1, j, -1):
            # calculate rotation matrix
            c, s = _givens(R[i - 1, j], R[i, j])
            # broadcast to an nxn matrix G
            G = np.identity(n)
            G[i - 1, i - 1] = c
            G[i, i] = c
            G[i, i - 1] = s
            G[i - 1, i] = -s
            # these matrices zero the lower diagonal elements of R
            # update R
            R = np.dot(G.T, R.copy())
            # update Q
            Q = np.dot(Q.copy(), G)
    return Q, R


def _givens_mat(c, s, n, i):
    '''
    Subroutine to return a givens rotation matrix

    Inputs- c, s (int): parameters for the givens matrix
            n (int): n_rows of the matrix
            i (int): index of the diagonal submatrix into which c and s should
                     be broadcast

    Output- G (np.array): givens rotation matrix
    '''
    # initialize matrix
    G = np.identity(n,)
    # assign values to the ith 2x2 diagonal submatrix
    G[i, i] = c
    G[i + 1, i + 1] = c
    G[i + 1, i] = s
    G[i, i + 1] = -s
    return G


def house_QR(A):
    '''
    Uses householder reflections to perform QR decomposition of A

    Input- A (np.array): the matrix to be decomposed

    Outputs- Q, R (np.array): an orthonormal matrix Q and an upper triangular
                              matrix R such that Q*R=A
    '''
    # get dimension of A
    n = len(A)
    # initialize Q
    Q = np.identity(n)
    # initialize R
    R = A.copy()
    # for n-1 columns
    for i in range(n - 1):
        # initialize householder tranformation
        H = np.identity(n)
        # embed reflection matrix
        H[i:, i:] = _house_mat(R[i:, i])
        # update R and Q
        Q = np.dot(Q, H)
        R = np.dot(H, R)
    return Q, R


def _house_mat(x):
    '''
    subroutine to calculate a householder reflection to zero all but the first
    entry of a column of a matrix

    Input- x (vector): vector to be reflected

    Output- H (matrix): householder transformation matrix. usually embedded into
                        an identity matrix
    '''
    # calculate normal to the hyperplane of reflection
    v = x / (x[0] + np.copysign(np.linalg.norm(x), x[0]))
    v[0] = 1
    # initialize H
    H = np.identity(x.shape[0])
    # calculate tranformation matrix
    H -= (2 / np.dot(v, v)) * np.outer(v, v)
    return H


def hessenberg(A, return_Q=True):
    '''
    Uses householder reflections to transform a matrix A into upper hessberg form

    Input- A (np.array): square matrix to be transformed

    Outputs- Q, H (np.array): orthonormal matrix Q and upper hessberg matrix H
                              such that Q*H=A
    '''
    # get dimension of A
    n = len(A)
    # initialize H
    R = A.copy()
    # initialize Q
    if return_Q:
        Q = np.identity(n)
    # perform n-2 householder reflections
    for i in range(n - 2):
        H = np.identity(n)
        # perform the reflection to zero all but the first 2 entries of the
        # column
        H[i + 1:, i + 1:] = _house_mat(R[i + 1:, i])
        R = H.dot(R).dot(H.T)
        if return_Q:
            Q = np.dot(Q, H)
    if return_Q:
        return Q, R
    else:
        return R


def hessenberg_QR(H):
    '''
    Uses givens rotations to perform QR decomposition on an upper hessberg matrix

    Input- H (np.array): square matrix in upper hessberg form

    Outputs- R (np.array): upper triangular matrix of the QR decomposition
             Gs (list): list of givens rotation matrices used to produce R
    '''
    # initialize R
    R = H.copy()
    # get dimension of H
    n = len(H)
    # initialize Gs
    Gs = []
    # perform n-1 givens rotations on the lower subdiagonal
    for i in range(n - 1):
        # get givens params
        c, s = _givens(R[i, i], R[i + 1, i])
        # get givens transformation matrix
        G = _givens_mat(c, s, n, i)
        # update R
        R = np.dot(G.T, R.copy())
        # update Gs
        Gs.append(G)
    return R, Gs


def eig(A, iterations=100):
    '''
    Uses givens rotations to perform the QR algorithm for finding real eigenvalues

    Inputs- A (np.array): matrix for which to find eigenvalues
            iterations (int): desired number of iterations

    Output- np.array: sorted array of real eigenvalues
    '''
    for _ in range(iterations):
        # calculate QR decomp
        Q, R = givens_QR(A)
        # compose new A
        A = R.dot(Q)
    # eigenvalues converge on the diagonal of A
    return np.sort(np.diag(A))


def hessenberg_eig(A, iterations=100):
    '''
    Converts a matrix into upper hessberg form, then uses givens rotations to
    perform the QR algorithm for finding real eigenvalues

    Inputs- A (np.array): matrix for which to find eigenvalues
            iterations (int): desired number of iterations

    Output- np.array: sorted array of real eigenvalues
    '''
    # convert A to upper hessberg form
    H = hessenberg(A, return_Q=False)
    for _ in range(iterations):
        # QR decompostion of H
        H, G = hessenberg_QR(H)
        # since Q is the product of all the transformations in G, R*Q can be
        # calculated as the product of R and all the transformations in G
        # this is guaranteed to return another upper hessberg matrix, for which
        # the QR decomposition can be computed again using the fast hessberg_QR
        # method
        for g in G:
            H = np.dot(H, g)
    # eigenvalues converge on the diagonal of H
    return np.sort(np.diag(H))

def eig_time(n, func, number=1000, repeats=3, A=None):
    '''
    Records running times for eigenvalue algorithms

    Inputs- n (int): size of test matrix
            func (function): eigenvalue function to be used
            num_trials (int): number of trials for the experiment
            repeats (int): number of experiments

    Output- list: times for each trial
    '''
    if A is None:
        # create random n x n matrix
        A = np.random.randint(0, n, (n, n))
        A = A.dot(A.T)

    def _timer():
        t1 = time.time()
        func(A)
        t2 = time.time()
        return t2 - t1
    # run all trials
    trials = [np.mean([_timer() for _ in range(number)])
              for _ in range(repeats)]
    return trials
