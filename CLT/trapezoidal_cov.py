import torch
from CLT.utils import apply_grid

def main_deteqv_cov(f_list,Gamma_list,L_diag_list,R=100,scalar1=1.5,scalar2=2):
    '''
    Numerical evaluation of deterministic equivalent covariance matrix Lambda_n.

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
    scalar1: float
        radius of contour 1 equal to scalar1 * minimum_possible_radius
    scalar2: float
        radius of contour 2 equal to scalar2 * minimum_possible_radius
    
    Returns 
    ----------
    cov: torch.Tensor
        numerically evaluated covariance matrix Lambda_n
    '''
    K=len(f_list)

    #L_squared_diag_list = [(v ** 2).to(dtype=torch.complex64) for v in L_diag_list]
    Gamma_list = [Gamma.to(dtype=torch.complex64) for Gamma in Gamma_list]
    L_diag_list = [L.to(dtype=torch.complex64) for L in L_diag_list]
    
    Gammas_tensor = torch.stack(Gamma_list)
    Ls_tensor = torch.stack(L_diag_list)
    #Ls_sq_tensor = torch.stack(L_squared_diag_list)

    cov=torch.zeros((K,K))
    sigma2=sigma_trap(Gammas_tensor,Ls_tensor,R=R,scalar1=scalar1,scalar2=scalar2)
    for r1 in range(K):
        for r2 in range(K):
            f=f_list[r1]
            g=f_list[r2]
            res=fg_trap(f,g,Gammas_tensor,Ls_tensor,R=R,scalar1=scalar1,scalar2=scalar2)
            cov[r1,r2]= torch.mean(res* sigma2)
    
    return cov.double()

def det_equiv_stieltjes(z,Gammas_tensor,Ls_tensor,thresh=0.001,iter_min=10):
    '''
    Numerical evaluation of the determinisitc equivalent Stieltjes transform \tilde{m}_n(z) 

    Parameters
	----------
    Gammas_tensor: torch.Tensor
        k_level_of_variations x n x n tensor of stacked Gamma matrices
    Ls_tensor: torch.Tensor
        k_level_of_variations x N tensor of stacked L vectors
    thresh: float
        convergence threshold of the iterative algorithm for evaluating the determinisitc equivalent Stieltjes transform
    iter_min: int
        minimum number of iterations

    Returns 
    ----------
    m: torch.Tensor
        Stieltjes transform of the deterministic equivalent law of the ESD of B_n
    b_tilde: torch.Tensor
        N-dimensional vector
    '''
    k = Ls_tensor.shape[0]
    n=Gammas_tensor.shape[-1]
    N=Ls_tensor.shape[-1]
    g1 = torch.ones(k)
    g2 = torch.ones(k)
    err = 1
    iter =0 
    while err>thresh or iter<iter_min:
        g1_reshaped = g1.view(-1, 1, 1)  
        Bn_det_eqv = (g1_reshaped * Gammas_tensor).sum(dim=0)*(-z)  
        identity = torch.eye(Bn_det_eqv.shape[0], dtype=torch.complex64)  
        g2_previous = g2.clone()
        g2 = torch.stack([
            torch.trace(torch.linalg.solve(Bn_det_eqv  - z * identity,Gammas_tensor[r]))/N
            for r in range(k)
        ])
        err = torch.max(torch.abs(g2 - g2_previous)).item()

        g2_reshaped = g2.view(-1, 1)  
        Bn_det_eqv = (g2_reshaped * Ls_tensor).sum(dim=0)*(-z)

        g1_previous = g1.clone()

        g1 = torch.stack([ 
            torch.sum(Ls_tensor[r,:]/(Bn_det_eqv  - z ))/N
            for r in range(k)
        ])    

        err = max(err,torch.max(torch.abs(g1 - g1_previous)).item())
        iter+=1

    # compute m as a function of g_2
    m=(N/n-1)/z+torch.sum(1/(Bn_det_eqv  - z ))/n 
    b_tilde = 1/(1+g2@Ls_tensor)

    return m, b_tilde

def Xi(z1,z2,Gammas_tensor,Ls_tensor,b_tilde_1,b_tilde_2):
    '''
    Evaluation of Xi^{ab}.

    Parameters
	----------
    Gammas_tensor: torch.Tensor
        k_level_of_variations x n x n tensor of stacked Gamma matrices
    Ls_tensor: torch.Tensor
        k_level_of_variations x N tensor of stacked L vectors.
    b_tilde_1: torch.Tensor
        N evaluations of b_tilde_i(z1) for i=1,...,N.
    b_tilde_2: torch.Tensor
        N evaluations of b_tilde_i(z2) for i=1,...,N.

    Returns 
    ----------
    res: torch.Tensor
        N x k x k evalations of h_j^{ab} for j=1,...,N, a=1,..,k, b=1,...,k.
    '''
    k = Ls_tensor.shape[0]
    n=Gammas_tensor.shape[-1]
    N=Ls_tensor.shape[-1]
    
    R_tilde_inv_1 = torch.linalg.inv(torch.einsum('i,ijk->jk', torch.matmul(Ls_tensor,b_tilde_1)/N, Gammas_tensor)-z1*torch.eye(n))
    R_tilde_inv_2 = torch.linalg.inv(torch.einsum('i,ijk->jk', torch.matmul(Ls_tensor,b_tilde_2)/N, Gammas_tensor)-z2*torch.eye(n))

    res=torch.zeros((k,k), dtype=torch.complex64)
    for a in range(k):
        for b in range(k):
            res[a,b] = torch.trace(R_tilde_inv_1@Gammas_tensor[a]@R_tilde_inv_2@Gammas_tensor[b])/N
    return res
    
def h(Ls_tensor,b_tilde_1,b_tilde_2):
    '''
    Evaluation of h_j^{ab}.

    Parameters
	----------
    Ls_tensor: torch.Tensor
        k_level_of_variations x N tensor of stacked L vectors.
    b_tilde_1: torch.Tensor
        N evaluations of b_tilde_i(z1) for i=1,...,N.
    b_tilde_2: torch.Tensor
        N evaluations of b_tilde_i(z2) for i=1,...,N.

    Returns 
    ----------
    res: torch.Tensor
        k x k evalations of h_j^{ab} for j=1,...,N, a=1,..,k, b=1,...,k.
    '''
    k = Ls_tensor.shape[0]
    N=Ls_tensor.shape[-1]
    b_mult = b_tilde_1*b_tilde_2
    res = torch.zeros((k, k), dtype=torch.complex64)  
    
    for a in range(k):
        for b in range(k):
            res[a, b] = torch.sum(Ls_tensor[a, :] * Ls_tensor[b, :] * b_mult) / N

    return res    

def sig2(z1,z2,Gammas_tensor,Ls_tensor):
    '''
    Evaluation of sigma_n^2(z_1,z_2).

    Parameters
	----------
    z1: torch.tensor
    Gammas_tensor: torch.Tensor
        k_level_of_variations x n x n tensor of stacked Gamma matrices
    Ls_tensor: torch.Tensor
        k_level_of_variations x N tensor of stacked L vectors.

    Returns 
    ----------
    res: torch.Tensor
        evaluation of sigma_n^2(z_1,z_2).
    '''
    if not isinstance(z1, torch.Tensor):
        z1 = torch.tensor(z1, dtype=torch.complex64, requires_grad=True)
    else:
        z1.requires_grad_(True)

    if not isinstance(z2, torch.Tensor):
        z2 = torch.tensor(z2, dtype=torch.complex64, requires_grad=True)
    else:
        z2.requires_grad_(True)

    k = Ls_tensor.shape[0]

    _, b_tilde_1 = det_equiv_stieltjes(z1,Gammas_tensor,Ls_tensor)
    _, b_tilde_2 = det_equiv_stieltjes(z2,Gammas_tensor,Ls_tensor)

    h_mat=h(Ls_tensor,b_tilde_1,b_tilde_2)
    Xi_mat = Xi(z1,z2,Gammas_tensor,Ls_tensor,b_tilde_1,b_tilde_2)
    Lambda_mat = h_mat@Xi_mat 
    sign, logabsdet = torch.linalg.slogdet(torch.eye(k)-Lambda_mat)
    res = -(torch.log(sign) + logabsdet)

    grad_z1_real = torch.autograd.grad(res.real, z1, create_graph=True)[0]
    grad_z1 = grad_z1_real.conj() 
    grad_z1_z2_real = torch.autograd.grad(grad_z1.real, z2)[0]
    grad_z1_z2 = grad_z1_z2_real.conj() 
    z1.requires_grad = False
    z2.requires_grad = False
    return grad_z1_z2


def sigma_trap(Gammas_tensor,Ls_tensor,R=100,scalar1=1.5,scalar2=2):
    '''
    Evaluation of sigma_n^2(z_1[i],z_2[j]) at each point on the trapezoidal rule grids,
    with z_1 being evaluation points on contour 1 and z_2 on contour 2.

    Parameters
	----------
    Gammas_tensor: torch.Tensor
        k_level_of_variations x n x n tensor of stacked Gamma matrices
    Ls_tensor: torch.Tensor
        k_level_of_variations x N tensor of stacked L vectors.
    R: int
        number of function evaluations in the trapezoidal rule
    scalar1 : float
        Radius of contour 1, set as scalar1 times the minimum valid radius.
    scalar2 : float
        Radius of contour 2, set as scalar2 times the minimum valid radius.

    Returns 
    ----------
    res: torch.tensor
        R x R evaluvations of sigma_n^2(z_1[i],z_2[j]) at each point of evaluation for the trapezoidal rule.
    '''
    k = Ls_tensor.shape[0]; n=Gammas_tensor.shape[-1]; N=Ls_tensor.shape[-1]
    C = n/N
    s_L = torch.sqrt(torch.max(Ls_tensor.double(), dim=1)[0])
    s_Gamma = torch.tensor([torch.sqrt(torch.linalg.svdvals(Gammas_tensor[r].double())[0]) for r in range(k)])
    imag = torch.tensor(complex(0,1))
    lam_min= 0 ; lam_max=((1+torch.sqrt(torch.tensor(C)))*torch.sum(s_L*s_Gamma))**2    
    scale_1 = (lam_max+lam_min)/2*scalar1;
    scale_2 = (lam_max+lam_min)/2*scalar2;
    displacement = (lam_max+lam_min)/2
    j_1 = torch.arange(1, R + 1)
    j_2 = torch.arange(1, R + 1)
    z_1= torch.exp(2*torch.pi*imag*j_1/R)
    z_2= torch.exp(2*torch.pi*imag*j_2/R)

    def sigma2_scaled(z_1_i, z_2_j):
        z1i = scale_1 * z_1_i + displacement
        z2j = scale_2 * z_2_j + displacement
        inte =sig2(z1i,z2j,Gammas_tensor,Ls_tensor)
        return inte.detach()
    
    res = apply_grid(z_1,z_2,sigma2_scaled,progress="Evaluating covariance...")
    res = res*2*scale_1*scale_2
    return res


def fg_trap(f,g,Gammas_tensor,Ls_tensor,R=100,scalar1=1.5,scalar2=2):
    '''
    Evaluation of f(z_1[i])g(z_2[j]) at each point on the trapezoidal rule grids,
    with z_1 being evaluation points on contour 1 and z_2 on contour 2.

    Parameters
	----------
    Gammas_tensor: torch.Tensor
        k_level_of_variations x n x n tensor of stacked Gamma matrices
    Ls_tensor: torch.Tensor
        k_level_of_variations x N tensor of stacked L vectors.
    R: int
        number of function evaluations in the trapezoidal rule
    scalar1 : float
        Radius of contour 1, set as scalar1 times the minimum valid radius.
    scalar2 : float
        Radius of contour 2, set as scalar2 times the minimum valid radius.

    Returns 
    ----------
    res: torch.tensor
        R x R evaluvations of f(z_1[i])g(z_2[j]) at each point of evaluation for the trapezoidal rule.
    '''
    k = Ls_tensor.shape[0]
    n=Gammas_tensor.shape[-1]
    N=Ls_tensor.shape[-1]
    C = n/N
    s_L = torch.sqrt(torch.max(Ls_tensor.double(), dim=1)[0])
    s_Gamma = torch.tensor([torch.sqrt(torch.linalg.svdvals(Gammas_tensor[r].double())[0]) for r in range(k)])

    imag = torch.tensor(complex(0,1))
    lam_min= 0 ; lam_max=((1+torch.sqrt(torch.tensor(C)))*torch.sum(s_L*s_Gamma))**2
    
    scale_1 = (lam_max+lam_min)/2*scalar1;
    scale_2 = (lam_max+lam_min)/2*scalar2;
    displacement = (lam_max+lam_min)/2
    
    j_1 = torch.arange(1,R+1)
    z_1= torch.exp(2*torch.pi*imag*j_1/R)
    j_2 = torch.arange(1,R+1)
    z_2= torch.exp(2*torch.pi*imag*j_2/R)
    def fg_scaled(z_1_i, z_2_j):
        z1i = scale_1 * z_1_i + displacement
        z2j = scale_2 * z_2_j + displacement
        num1 = f(z1i)
        num2 = g(z2j) 
        inte =num1*num2*z_1_i*z_2_j
        return inte.detach()
    result = apply_grid(z_1,z_2,fg_scaled)
    return result


