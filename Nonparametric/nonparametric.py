import numpy as np
from tqdm import tqdm
from scipy import optimize
from sklearn.isotonic import IsotonicRegression

from Nonparametric.kernel import select_eps_cv,kernel_density

def det_equiv_stieltjes(z,Sigma_list,U_list,B,thresh=0.001):
    k = len(Sigma_list)
    p = Sigma_list[0].shape[0]
    n = B.shape[0]
    err = 1
    b = np.zeros(k, dtype=complex)
    while err>thresh:
        a_tilde = np.zeros(k, dtype=complex)
        M = z * np.eye(p)
        for r in range(k):
            M = M + b[r] * Sigma_list[r]
        M_inv = np.linalg.inv(M)
        for r in range(k):
            a_tilde[r] = -(1.0 / p) * np.trace(M_inv @ Sigma_list[r])
        b_new = np.zeros(k, dtype=complex)
        M = np.eye(n)
        for r in range(k):
            M = M + a_tilde[r] * U_list[r]@U_list[r].T@B
        M_inv = np.linalg.inv(M)
        for r in range(k):
            b_new[r] = -(1.0 / p) * np.trace(M_inv @ U_list[r]@U_list[r].T@B)
        err =  np.max(np.abs(b_new - b))
        b = b_new
    a_tilde = np.zeros(k, dtype=complex)
    M = z * np.eye(p)
    for r in range(k):
        M = M + b[r] * Sigma_list[r]
    M_inv = np.linalg.inv(M)
    for r in range(k):
        a_tilde[r] = -(1.0 / p) * np.trace(M_inv @ Sigma_list[r])
    m_tilde =  -(1.0 / p) * np.trace(M_inv)
    return m_tilde,a_tilde,b

def deteqv_oracle_est(Y,Sigma_list,U_list,B,thresh=0.001):
    """
    Computes deterministic equivalent oracle estimator for Sigma_1.
    """
    p = Sigma_list[0].shape[0]
    Sn = 1/p*Y.T@B@Y
    Lambda, _ = np.linalg.eig(Sn.astype(np.float64))
    isort = np.argsort(Lambda)[::-1]
    Lambda = Lambda[isort]

    d_check = np.zeros(p, dtype=float)
    m_check = np.zeros(p, dtype=complex)
    a1_check = np.zeros(p, dtype=complex)

    for i in tqdm(range(p),desc="Calculating deterministic equivalent oracle..."):    
        z = complex(Lambda[i], 1e-3)
        m0,a_check,_=det_equiv_stieltjes(z,Sigma_list,U_list,B,thresh=thresh)
        m_check[i] = m0
        a1_check[i] = a_check[0]
        d_check[i] = np.imag(a_check[0]) / np.imag(m0)
    return d_check,m_check.imag,a1_check.imag

def bona_fide_est(Y,U_list,B, thresh = 1e-5, isotonic=True,seed = 1):
    """
    Computes bona fide nonparametric estimator for Sigma_1.
    """
    np.random.seed(seed)
    k = len(U_list)
    p = Y.shape[1]
    n = Y.shape[0]
    I_list = [Ur.shape[1] for Ur in U_list]

    Sn = 1/p*Y.T@B@Y
    Lambda, U = np.linalg.eig(Sn.astype(np.float64))
    isort = np.argsort(Lambda)[::-1]
    Lambda = Lambda[isort]
    U = U[:, isort]
    U_vec_F = np.hstack([np.sqrt(Ur.shape[1])*Ur for Ur in U_list])
    F = 1/p*U_vec_F.T@B@U_vec_F
    T_list = [1/p*np.diag(U.T@Y.T@B@Ur@Ur.T@B@Y@U) for Ur in U_list]


    ## kernel smoothing
    eps_cv = select_eps_cv(Lambda, n)

    f_hat,_ = kernel_density(Lambda,n,eps=eps_cv)
    g_hat_list = []; Hg_hat_list=[]
    for r in range(k):
        g_hat_r,Hg_hat_r = kernel_density(Lambda,n,T=T_list[r],eps=eps_cv)
        g_hat_list.append(g_hat_r)
        Hg_hat_list.append(Hg_hat_r)

    ## bona fide nonparametric estimate
    temp = -np.diag(F)
    correction = [np.mean(temp[int(np.sum(I_list[:r])):int(np.sum(I_list[:r+1]))]) for r in range(k)]
    b_hat_list = []

    for r in range(k):
        br_hat = np.array([correction[r]+np.pi*complex(Hg_hat_list[r][i],g_hat_list[r][i]) for i in range(p)], dtype=np.complex128)
        b_hat_list.append(br_hat)    

    def a_to_b(a):
        b = np.zeros_like(a)
        M = np.eye(n)
        for r in range(k):
            M = M + a[r] * U_list[r]@U_list[r].T@B
        M_inv = np.linalg.inv(M)
        for r in range(k):
            b[r] = -(1.0 / p) * np.trace(M_inv @ U_list[r]@U_list[r].T@B)
        return b

    def a_to_b_func(a_flat):
        a = np.array([complex(a_flat[2*i],a_flat[2*i+1]) for i in range(int(len(a_flat)/2))])
        computed_b = a_to_b(a)
        return np.array([x for c in computed_b for x in (c.real, c.imag)])
    
    d_hat = np.zeros(p)
    a1_hat_Im = np.zeros(p)
    for i in tqdm(range(p),desc="computing a_hat..."):
        target_b =  np.concatenate([[b_hat_list[r][i].real, b_hat_list[r][i].imag] for r in range(k)])
        a_flat_hat = np.zeros(2*k) 
        error = np.linalg.norm(a_to_b_func(a_flat_hat) - target_b)

        def solve_equation():
            a_flat_hat = optimize.fsolve(lambda  x: a_to_b_func(x) - target_b,np.random.uniform(0,1,len(target_b)),maxfev=5000)#
            return a_flat_hat
        
        error = np.linalg.norm(a_to_b_func(a_flat_hat) - target_b)
        while error > thresh:
            a_flat_hat = solve_equation()
            error = np.linalg.norm(a_to_b_func(a_flat_hat) - target_b)
        d_hat[i] = a_flat_hat[1] / f_hat[i] / np.pi 
        a1_hat_Im[i] = a_flat_hat[1]

    m_hat_Im = f_hat*np.pi
    
    ## isotonising post processing step
    if isotonic:
        iso = IsotonicRegression(increasing=False)
        d_iso = iso.fit_transform(np.arange(1, len(d_hat)+1), d_hat)
        return d_hat, d_iso, m_hat_Im, a1_hat_Im
    return d_hat, m_hat_Im, a1_hat_Im
