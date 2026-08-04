import numpy as np
from numpy.linalg import matrix_power

def matrix_mmts(M, order):
    '''
    Return empirical mmts M^k for k = 1, ..., order.
    '''
    return np.array([np.trace(matrix_power(M, k))/M.shape[0] for k in range(1, order + 1)])

def generate_owub_design(I, J_list=None, include_intercept=True, seed=1):
    '''
    One-way unbalanced (full-sib) design.

    Family sizes J_1,...,J_I are i.i.d. Unif{1,2}.    

    Parameters
    ----------
    include_intercept : bool
    If True, fixed effect is X = 1_n. If False, X = 0.
    '''
    np.random.seed(seed)
    k = 2
    if J_list is None:
        J_list = np.random.binomial(size = I, n = 1, p = 0.5) + 1

    n=np.sum(J_list)
    
    U0 = np.ones((n, 1))
    
    U1=np.zeros([n,I])
    sum_=0
    for i in range(I):
        U1[sum_:(sum_+J_list[i]),i]=1
        sum_=sum_+J_list[i]
                
    U2= np.eye(n)
    
    U_list = [U0,U1,U2]
    
    B_list = []
    
    for i in range(k):
        if i==0 and not include_intercept:
            B_list.append(U_list[1]@np.linalg.inv(U_list[1].T@U_list[1])@U_list[1].T)
            continue
        B_list.append(U_list[i+1]@np.linalg.inv(U_list[i+1].T@U_list[i+1])@U_list[i+1].T 
                      - U_list[i]@np.linalg.inv(U_list[i].T@U_list[i])@U_list[i].T)
    return U_list[1:], B_list, J_list, n


def generate_twub_design(I_1, M, J=[], J2=[],seed=1):
    '''
    Two-way unbalanced (full-sib half-sib) design with I_1 sires and M dams.

    Family sizes J_{im} corresponding to the i-th sire and m-th dam are i.i.d. Unif{1,2}.    
    '''
    np.random.seed(seed)
    k = 3
    I_2 = I_1*M

    if len(J)==0:
        J2 = np.random.binomial(1, 0.5, size=(I_1, M))+1
        J = np.sum(J2,axis=1)    
        J2 = J2.reshape(-1)
    
    n=np.sum(J); 
    
    X = np.ones((n, 1))
    
    U1=np.zeros([n,I_1])
    sum_=0
    for i in range(I_1):
        U1[sum_:(sum_+J[i]),i]=1
        sum_=sum_+J[i]
                
    U2=np.zeros([n,len(J2)])
    sum_=0        
    for i in range(len(J2)):
        U2[sum_:(sum_+J2[i]),i]=1
        sum_=sum_+J2[i]
    
    U3 = np.eye(n)
    
    U_list = [U1,U2,U3]

    B_list = unbalanced_nested_projections(X, U_list)

    return U_list, B_list, J, J2, n

def orth(A, tol=1e-10):
    """Orthonormal basis for col(A)."""
    if A.size == 0:
        return np.zeros((A.shape[0], 0))
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    rank = np.sum(s > tol * max(A.shape) * (s[0] if s.size else 1.0))
    return U[:, :rank]

def nullspace(A, tol=1e-10):
    """Orthonormal basis for ker(A)."""
    if A.size == 0:
        return np.eye(A.shape[1])
    _, s, Vt = np.linalg.svd(A, full_matrices=True)
    scale = s[0] if s.size else 1.0
    rank = np.sum(s > tol * max(A.shape) * scale)
    return Vt[rank:].T

def unbalanced_nested_projections(X, U_list, tol=1e-10):
    """
    Greedy top-down construction of projection matrices B_r.

    Inputs:
        X: n x I0 matrix
        U_list: [U1, ..., Uk], each n x Ir

    Returns:
        B_list: [B1, ..., Bk]
        Q_list: orthonormal bases [Q1, ..., Qk] for W_r
        E_list: orthonormal bases for standard increment spaces
    """
    k = len(U_list)
    n = X.shape[0]
    I = np.eye(n)

    QV = [orth(X, tol)]
    for U in U_list:
        QV.append(orth(U, tol))

    # Bases E_r 
    E = []
    for r in range(k ):
        U_r = U_list[r ]
        Q_prev = QV[r ]
        U_res = (I - Q_prev @ Q_prev.T) @ U_r
        E.append(orth(U_res, tol))

    Q = [None] * k
    B = [None] * k

    # Top level: keep the full increment space
    Q[k - 1] = E[k - 1]
    B[k - 1] = Q[k - 1] @ Q[k - 1].T
    print(f"[level {k}] increment space dimension: ", Q[k - 1].shape[1],f", rank of B_{k}: ", Q[k - 1].shape[1])

    # Work downward
    for r in range(k - 2, -1, -1):  # r = k-1,...,1 in 1-based indexing
        E_r = E[r]
        d_r = E_r.shape[1]

        if d_r == 0:
            Q[r ] = np.zeros((n, 0))
            B[r ] = np.zeros((n, n))
            print("Failed!")
            continue

        blocks = []
        for s in range(r + 1, k ):      # s > r
            Q_s = Q[s ]
            if Q_s.shape[1] == 0:
                print("Failed!")
                continue
            for t in range(s, k ):      # t >= s
                U_t = U_list[t ]
                blocks.append(Q_s.T @ U_t @ U_t.T @ E_r)

        if blocks:
            M = np.vstack(blocks)
            M[np.abs(M) < tol] = 0.0
        else:
            M = np.zeros((0, d_r))

        N_r = nullspace(M, tol)
        Q_r = orth(E_r @ N_r, tol)

        Q[r ] = Q_r
        B[r ] = Q_r @ Q_r.T
        print(f"[level {r+1}] increment space dimension: ", E_r.shape[1],f", rank of B_{r+1}: ", Q_r.shape[1])

    return B

def check_conditions(X, U_list, B_list):
    """Return max Frobenius norm violations of the desired identities."""
    k = len(U_list)

    max_BX = 0.0
    max_lower = 0.0
    max_cross = 0.0

    for r in range( k ):
        B_r = B_list[r ]
        max_BX = max(max_BX, np.linalg.norm(B_r @ X, ord="fro"))

        for s in range( r):
            U_s = U_list[s ]
            max_lower = max(max_lower, np.linalg.norm(B_r @ U_s, ord="fro"))

        for s in range(k):
            if s == r:
                continue
            B_s = B_list[s ]
            for t in range(max(r, s), k ):
                U_t = U_list[t ]
                val = B_r @ U_t @ U_t.T @ B_s
                max_cross = max(max_cross, np.linalg.norm(val, ord="fro"))

    return {
        "max ||B_r X||_F": max_BX,
        "max ||B_r U_s||_F for s<r": max_lower,
        "max ||B_r U_t U_t^T B_s||_F": max_cross,
    }

