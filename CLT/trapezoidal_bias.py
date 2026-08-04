import torch
from CLT.utils import apply_vec
from CLT.trapezoidal_cov import h,det_equiv_stieltjes

def main_deteqv_bias(f_list,Gamma_list,L_diag_list,R=100,scalar=2):
    '''
    Numerical evaluation of deterministic equivalent bias vector Gamma_n

    Parameters
	----------
    f_list: list of callables
        functions [f_1,...,f_K]
    Gamma_list: list of np.ndarray or torch.Tensor matrices
        Gamma matrices [Gamma_1,...,Gamma_k]
    L_diag_list: list of np.ndarray or torch.Tensor vectors
        L vectors [L_1,...,L_k]
    R: int
        number of function evaluations in the trapezoidal rule
    scalar: float
        radius of contour equal to scalar * minimum_possible_radius
    
    Returns 
    ----------
    bias: torch.Tensor
        numerically evaluated bias vector Gamma_n
    '''
    K= len(f_list)
    Gamma_list = [Gamma.to(dtype=torch.complex64) for Gamma in Gamma_list]
    L_list = [L.to(dtype=torch.complex64) for L in L_diag_list]
    Gammas_tensor = torch.stack(Gamma_list)
    Ls_tensor = torch.stack(L_list)
    
    bias=torch.zeros(K, dtype=torch.complex64)
    mu=mu_trap(Gammas_tensor,Ls_tensor,R=R,scalar=scalar)
    for r in range(K):
        f = f_list[r]
        res=f_trap(f,Gammas_tensor,Ls_tensor,R=R,scalar=scalar)
        bias[r]= torch.mean(res* mu)
    return bias.double()

def main_deteqv_lss(f_list,Gamma_list,L_diag_list,R=100,scalar=2):
    '''
    Numerical evaluation of deterministic equivalent LSS

    Parameters
	----------
    f_list: list of callables
        functions [f_1,...,f_K]
    Gamma_list: list of np.array or torch.Tensor matrices
        Gamma matrices [Gamma_1,...,Gamma_k]
    L_diag_list: list of np.array or torch.Tensor vectors
        L vectors [L_1,...,L_k]
    R: int
        number of function evaluations in the trapezoidal rule
    scalar: float
        radius of contour equal to scalar * minimum_possible_radius
    
    Returns 
    ----------
    deteqvs: torch.Tensor
        numerically evaluated deterministic equivalent LSS
    '''
    K= len(f_list)
    k = len(Gamma_list)
    n = Gamma_list[0].shape[0]
    N = L_diag_list[0].shape[0]

    Gamma_list = [Gamma.to(dtype=torch.complex64) for Gamma in Gamma_list]
    L_diag_list = [L.to(dtype=torch.complex64) for L in L_diag_list]
    Gammas_tensor = torch.stack(Gamma_list)
    Ls_tensor = torch.stack(L_diag_list)

    C = n/N
    s_L = torch.sqrt(torch.max(Ls_tensor.double(), dim=1)[0])
    s_Gamma = torch.tensor([torch.sqrt(torch.linalg.svdvals(Gammas_tensor[r].double())[0]) for r in range(k)])

    imag = torch.tensor(complex(0,1))
    lam_min= 0 ; lam_max=((1+torch.sqrt(torch.tensor(C)))*torch.sum(s_L*s_Gamma))**2
    
    scale = (lam_max+lam_min)/2*scalar;
    displacement = (lam_max+lam_min)/2
    
    j = torch.arange(1,R+1)
    z= torch.exp(2*torch.pi*imag*j/R)

    def stieltjes_scaled(z_i):
        zi = scale * z_i + displacement
        m, _ = det_equiv_stieltjes(zi,Gammas_tensor,Ls_tensor)
        inte =m
        return -inte.detach()*scale
    result = apply_vec(z,stieltjes_scaled,progress="Evaluating deterministic equivalent LSS...")
    deteqvs=torch.zeros(K)
    for r in range(K):
        f=f_list[r]
        res=f_trap(f,Gammas_tensor,Ls_tensor,R=R,scalar=scalar)
        deteqvs[r]=torch.mean(res*result)
    return deteqvs

def Xi_0_a(z,Gammas_tensor,Ls_tensor,b_tilde):
    '''
    Evaluation of Xi_0^{a}.

    Parameters
	----------
    Gammas_tensor: torch.Tensor
        k_level_of_variations x n x n tensor of stacked Gamma matrices
    Ls_tensor: torch.Tensor
        k_level_of_variations x N tensor of stacked L vectors.
    b_tilde: torch.Tensor
        N evaluations of b_tilde_i(z) for i=1,...,N.

    Returns 
    ----------
    res: torch.Tensor
        k evalations of Xi_0^{a} for a=1,..,k.
    '''
    k = Ls_tensor.shape[0]
    n=Gammas_tensor.shape[-1]
    N=Ls_tensor.shape[-1]
    
    R_tilde_inv = torch.linalg.inv(torch.einsum('i,ijk->jk', torch.matmul(Ls_tensor,b_tilde)/N, Gammas_tensor)-z*torch.eye(n))

    res=torch.zeros(k, dtype=torch.complex64)
    for a in range(k):
        res[a] = torch.trace(R_tilde_inv@Gammas_tensor[a]@R_tilde_inv)/N
    return res

def Xi_1_ab(z,Gammas_tensor,Ls_tensor,b_tilde):
    '''
    Evaluation of Xi_1^{ab}.

    Parameters
	----------
    Gammas_tensor: torch.Tensor
        k_level_of_variations x n x n tensor of stacked Gamma matrices
    Ls_tensor: torch.Tensor
        k_level_of_variations x N tensor of stacked L vectors.
    b_tilde: torch.Tensor
        N evaluations of b_tilde_i(z) for i=1,...,N.

    Returns 
    ----------
    res: torch.Tensor
        k x k evalations of Xi_1^{ab} for a=1,..,k, b=1,...,k.
    '''
    k = Ls_tensor.shape[0]
    n=Gammas_tensor.shape[-1]
    N=Ls_tensor.shape[-1]
    
    R_tilde_inv = torch.linalg.inv(torch.einsum('i,ijk->jk', torch.matmul(Ls_tensor,b_tilde)/N, Gammas_tensor)-z*torch.eye(n))

    res=torch.zeros((k,k), dtype=torch.complex64)
    for a in range(k):
        for b in range(k):
            res[a,b] = torch.trace(R_tilde_inv@Gammas_tensor[a]@R_tilde_inv@Gammas_tensor[b])/N
    return res

def Xi_2_ab(z,Gammas_tensor,Ls_tensor,b_tilde):
    '''
    Evaluation of Xi_1^{ab}.

    Parameters
	----------
    Gammas_tensor: torch.Tensor
        k_level_of_variations x n x n tensor of stacked Gamma matrices
    Ls_tensor: torch.Tensor
        k_level_of_variations x N tensor of stacked L vectors.
    b_tilde: torch.Tensor
        N evaluations of b_tilde_i(z) for i=1,...,N.

    Returns 
    ----------
    res: torch.Tensor
        k x k evalations of Xi_1^{ab} for a=1,..,k, b=1,...,k.
    '''
    k = Ls_tensor.shape[0]
    n=Gammas_tensor.shape[-1]
    N=Ls_tensor.shape[-1]
    
    R_tilde_inv = torch.linalg.inv(torch.einsum('i,ijk->jk', torch.matmul(Ls_tensor,b_tilde)/N, Gammas_tensor)-z*torch.eye(n))

    res=torch.zeros((k,k), dtype=torch.complex64)
    for a in range(k):
        for b in range(k):
            res[a,b] = torch.trace(R_tilde_inv@Gammas_tensor[a]@R_tilde_inv@R_tilde_inv@Gammas_tensor[b])/N
    return res

def Xi_3_abc(z,Gammas_tensor,Ls_tensor,b_tilde):
    '''
    Evaluation of Xi_3^{abc}.

    Parameters
	----------
    Gammas_tensor: torch.Tensor
        k_level_of_variations x n x n tensor of stacked Gamma matrices
    Ls_tensor: torch.Tensor
        k_level_of_variations x N tensor of stacked L vectors.
    b_tilde: torch.Tensor
        N evaluations of b_tilde_i(z) for i=1,...,N.

    Returns 
    ----------
    res: torch.Tensor
        k x k x k evalations of Xi_1^{abc} for a=1,..,k, b=1,...,k, c=1,...,k.
    '''
    k = Ls_tensor.shape[0]
    n=Gammas_tensor.shape[-1]
    N=Ls_tensor.shape[-1]
    
    R_tilde_inv = torch.linalg.inv(torch.einsum('i,ijk->jk', torch.matmul(Ls_tensor,b_tilde)/N, Gammas_tensor)-z*torch.eye(n))

    res=torch.zeros((k,k,k), dtype=torch.complex64)
    for a in range(k):
        for b in range(k):
            for c in range(k):
                res[a,b,c] = torch.trace(R_tilde_inv@Gammas_tensor[a]@R_tilde_inv@Gammas_tensor[b]@R_tilde_inv@Gammas_tensor[c])/N
    return res

def h_abc(Ls_tensor,b_tilde):
    '''
    Evaluation of h^{abc}.

    Parameters
	----------
    Gammas_tensor: torch.Tensor
        k_level_of_variations x n x n tensor of stacked Gamma matrices
    Ls_tensor: torch.Tensor
        k_level_of_variations x N tensor of stacked L vectors.
    b_tilde: torch.Tensor
        N evaluations of b_tilde_i(z) for i=1,...,N.

    Returns 
    ----------
    res: torch.Tensor
        k x k x k evalations of h^{abc} for a=1,..,k, b=1,...,k, c=1,...,k.
    '''
    k = Ls_tensor.shape[0]
    b_mult = b_tilde**3
    res=torch.zeros((k,k,k), dtype=torch.complex64)
    for a in range(k):
        for b in range(k):
            for c in range(k):
                res[a,b,c] = torch.mean(Ls_tensor[a,:]*Ls_tensor[b,:]*Ls_tensor[c,:]*b_mult)
    return res 



def zetas(z,Gammas_tensor,Ls_tensor,b_tilde):
    '''
    Evaluation of zeta_1^{ab}, zeta_2^{ab}, zeta_3^{abc}.

    Parameters
	----------
    Gammas_tensor: torch.Tensor
        k_level_of_variations x n x n tensor of stacked Gamma matrices
    Ls_tensor: torch.Tensor
        k_level_of_variations x N tensor of stacked L vectors.
    b_tilde: torch.Tensor
        N evaluations of b_tilde_i(z) for i=1,...,N.

    Returns 
    ----------
    zeta_1_mat: torch.Tensor
        k x k evalations of zeta_1^{ab} for a=1,..,k, b=1,...,k.
    zeta_2_mat: torch.Tensor
        k x k evalations of zeta_2^{ab} for a=1,..,k, b=1,...,k.
    zeta_3_mat: torch.Tensor
        k x k x k evalations of zeta_3^{abc} for a=1,..,k, b=1,...,k, c=1,...,k.
    '''
    k = Ls_tensor.shape[0]
    Xi_1_mat =  Xi_1_ab(z,Gammas_tensor,Ls_tensor,b_tilde)
    h_N_plus_1_mat = h(Ls_tensor,b_tilde,b_tilde)
    multiplier_mat = torch.eye(k) - h_N_plus_1_mat@Xi_1_mat.T
    
    zeta_1_mat= torch.linalg.solve(multiplier_mat,Xi_1_mat)

    Xi_2_mat = Xi_2_ab(z,Gammas_tensor,Ls_tensor,b_tilde)
    multiplier_mat_2= zeta_1_mat@h_N_plus_1_mat@Xi_2_mat.T

    zeta_2_mat = Xi_2_mat + multiplier_mat_2

    Xi_3_mat = Xi_3_abc(z,Gammas_tensor,Ls_tensor,b_tilde)
    multiplier_mat_3= torch.einsum('ij,abj->abi', zeta_1_mat@h_N_plus_1_mat, Xi_3_mat.permute(2, 1, 0)).permute(2, 1, 0)
    zeta_3_mat = Xi_3_mat + multiplier_mat_3
    
    return zeta_1_mat,zeta_2_mat,zeta_3_mat

def d_tilde(z,Gammas_tensor,Ls_tensor,b_tilde):
    '''
    Evaluation of \tilde{d}_{n0},\tilde{d}_{n1},...,\tilde{d}_{nk}.

    Parameters
	----------
    Gammas_tensor: torch.Tensor
        k_level_of_variations x n x n tensor of stacked Gamma matrices
    Ls_tensor: torch.Tensor
        k_level_of_variations x N tensor of stacked L vectors.
    b_tilde: torch.Tensor
        N evaluations of b_tilde_i(z) for i=1,...,N.

    Returns 
    ----------
    d_tilde_vec: torch.Tensor
        tensor of length k+1 containing evalations of \tilde{d}_{n0},\tilde{d}_{n1},...,\tilde{d}_{nk}.
    '''
    k = Ls_tensor.shape[0]
    h_mat = h_abc(Ls_tensor,b_tilde)
    h_N_plus_1_mat = h(Ls_tensor,b_tilde,b_tilde)
    
    Xi_0_mat =  Xi_0_a(z,Gammas_tensor,Ls_tensor,b_tilde)
    Xi_1_mat =  Xi_1_ab(z,Gammas_tensor,Ls_tensor,b_tilde)
    
    zeta_1_mat,zeta_2_mat,zeta_3_mat = zetas(z,Gammas_tensor,Ls_tensor,b_tilde)

    d_tilde_vec = torch.zeros(k+1, dtype=torch.complex64)
    d_tilde_vec[0] = torch.sum(h_N_plus_1_mat*zeta_2_mat) - torch.sum(torch.einsum('abc,c->ab', h_mat, Xi_0_mat)*zeta_1_mat)
    for r in range(k):
        d_tilde_vec[r+1] = torch.sum(h_N_plus_1_mat*zeta_3_mat[:,:,r])-torch.sum(torch.einsum('abc,c->ab', h_mat, Xi_1_mat[:,r])*zeta_1_mat)
    
    return d_tilde_vec

def nu(z,Gammas_tensor,Ls_tensor,b_tilde):
    '''
    Evaluation of nu_1,...,nu_k.

    Parameters
	----------
    Gammas_tensor: torch.Tensor
        k_level_of_variations x n x n tensor of stacked Gamma matrices
    Ls_tensor: torch.Tensor
        k_level_of_variations x N tensor of stacked L vectors.
    b_tilde: torch.Tensor
        N evaluations of b_tilde_i(z) for i=1,...,N.

    Returns 
    ----------
    res: torch.Tensor
        tensor of length k containing evalations of nu_1,...,nu_k.
    '''
    k = Ls_tensor.shape[0]
    n=Gammas_tensor.shape[-1]
    N=Ls_tensor.shape[-1]
    h_N_plus_1_mat = h(Ls_tensor,b_tilde,b_tilde)
    Xi_1_mat =  Xi_1_ab(z,Gammas_tensor,Ls_tensor,b_tilde)
    d_tilde_vec = d_tilde(z,Gammas_tensor,Ls_tensor,b_tilde)
    res = torch.linalg.solve(torch.eye(k)-Xi_1_mat.T@h_N_plus_1_mat,n/N*d_tilde_vec[1:]) 
    return res


def mu(z,Gammas_tensor,Ls_tensor,thresh=0.001):
    '''
    Evaluation of of mu_n(z).

    Parameters
	----------
    Gammas_tensor: torch.Tensor
        k_level_of_variations x n x n tensor of stacked Gamma matrices
    Ls_tensor: torch.Tensor
        k_level_of_variations x N tensor of stacked L vectors.
    b_tilde: torch.Tensor
        N evaluations of b_tilde_i(z) for i=1,...,N.
    thresh: float
        convergence threshold of the iterative algorithm for evaluating the determinisitc equivalent Stieltjes transform

    Returns 
    ----------
    res: torch.Tensor
        evaluation of mu_n(z).
    '''
    n=Gammas_tensor.shape[-1]
    N=Ls_tensor.shape[-1]
    
    _, b_tilde = det_equiv_stieltjes(z,Gammas_tensor,Ls_tensor,thresh=thresh)
    h_N_plus_1_mat = h(Ls_tensor,b_tilde,b_tilde)
    Xi_0_mat =  Xi_0_a(z,Gammas_tensor,Ls_tensor,b_tilde)
    nu_vec = nu(z,Gammas_tensor,Ls_tensor,b_tilde)
    d_tilde_vec = d_tilde(z,Gammas_tensor,Ls_tensor,b_tilde)

    res = d_tilde_vec[0]+N/n* nu_vec.T@ (h_N_plus_1_mat@Xi_0_mat)
    return res


def mu_trap(Gammas_tensor,Ls_tensor,R=100,scalar=2):
    '''
    Evaluation of mu_n(z[i]) at each contour point used in the trapezoidal rule.

    Parameters
	----------
    Gammas_tensor: torch.Tensor
        k_level_of_variations x n x n tensor of stacked Gamma matrices
    Ls_tensor: torch.Tensor
        k_level_of_variations x N tensor of stacked L vectors.
    R: int
        number of function evaluations in the trapezoidal rule
    scalar : float
        Radius of contour, set as scalar times the minimum valid radius.

    Returns 
    ----------
    res: torch.tensor
        R evaluvations of mu_n^2(z[i]) at each point of evaluation used in the trapezoidal rule.
    '''
    k = Ls_tensor.shape[0]; n=Gammas_tensor.shape[-1]; N=Ls_tensor.shape[-1]
    C = n/N
    s_L = torch.sqrt(torch.max(Ls_tensor.double(), dim=1)[0])
    s_Gamma = torch.tensor([torch.sqrt(torch.linalg.svdvals(Gammas_tensor[r].double())[0]) for r in range(k)])
    imag = torch.tensor(complex(0,1))
    lam_min= 0 ; lam_max=((1+torch.sqrt(torch.tensor(C)))*torch.sum(s_L*s_Gamma))**2
    scale = (lam_max+lam_min)/2*scalar;
    displacement = (lam_max+lam_min)/2
    j = torch.arange(1,R+1)
    z= torch.exp(2*torch.pi*imag*j/R)

    def mu_scaled(z_i):
        zi = scale * z_i + displacement
        inte =mu(zi,Gammas_tensor,Ls_tensor)
        return inte.detach()
    
    result = apply_vec(z,mu_scaled,progress="Evaluating bias...")
    return -result*scale

def f_trap(f,Gammas_tensor,Ls_tensor,R=100,scalar=2):
    '''
    Evaluation of f(z[i]) at each contour point used in the trapezoidal rule.

    Parameters
	----------
    Gammas_tensor: torch.Tensor
        k_level_of_variations x n x n tensor of stacked Gamma matrices
    Ls_tensor: torch.Tensor
        k_level_of_variations x N tensor of stacked L vectors.
    R: int
        number of function evaluations in the trapezoidal rule
    scalar : float
        Radius of contour, set as scalar times the minimum valid radius.

    Returns 
    ----------
    res: torch.tensor
        R evaluvations of f(z[i]) at each point of evaluation for the trapezoidal rule.
    '''
    k = Ls_tensor.shape[0]
    n=Gammas_tensor.shape[-1]
    N=Ls_tensor.shape[-1]
    C = n/N
    s_L = torch.sqrt(torch.max(Ls_tensor.double(), dim=1)[0])
    s_Gamma = torch.tensor([torch.sqrt(torch.linalg.svdvals(Gammas_tensor[r].double())[0]) for r in range(k)])

    imag = torch.tensor(complex(0,1))
    lam_min= 0 ; lam_max=((1+torch.sqrt(torch.tensor(C)))*torch.sum(s_L*s_Gamma))**2
    
    scale = (lam_max+lam_min)/2*scalar;
    displacement = (lam_max+lam_min)/2
    
    j = torch.arange(1,R+1)
    z= torch.exp(2*torch.pi*imag*j/R)
    def f_scaled(z_i):
        zi = scale * z_i + displacement
        num = f(zi)
        inte =num*z_i
        return inte.detach()
    result = apply_vec(z,f_scaled)
    return result



