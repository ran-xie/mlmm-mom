import torch
from CLT.trapezoidal_cov import apply_grid,det_equiv_stieltjes
from CLT.trapezoidal_bias import apply_vec

def wishart_deteqv_bias(f_list,N,Sigma_n,R=100,scalar=2,exclude_origin=False):
    '''
    Numerical evaluation of deterministic equivalent bias vector Gamma_n 
    specialized to classical Wishart matrices Bn = (1/N)Sigma_n^{1/2}X_nX_n^TSigma_n^{1/2}.

    Parameters
	----------
    f_list: list of callables
        functions [f_1,...,f_K]
    N: int
        number of columns in X_n
    Sigma_n: np.array or torch.Tensor
        population covariance matrix of the Wishart matrix
    R: int
        number of function evaluations in the trapezoidal rule
    exclude_origin: bool
        if the contour excludes the origin 
    scalar: float
        Must be > 1.
        - if exclude_origin = False: radius of contour equal to scalar * minimum_possible_radius
        - if exclude_orign = True: radius of contour equal to 1/scalar * maximum_possible_radius
    
    Returns 
    ----------
    bias: torch.Tensor
        numerically evaluated bias vector Gamma_n
    '''
    K = len(f_list)
    n = Sigma_n.shape[0]
    c_n = n/N

    Sigma_list = [Sigma_n.to(dtype=torch.complex64)]
    Sigmas_tensor = torch.stack(Sigma_list)
    Ls_tensor = torch.stack([(torch.ones(N)).to(dtype=torch.complex64)])
    
    bias = torch.zeros(K, dtype=torch.complex64)
    
    mu = mu_trap_wishart(Sigmas_tensor, Ls_tensor, R = R, scalar = scalar, exclude_origin = exclude_origin)

    for r in range(K):
        f=f_list[r]
        res=f_trap_wishart(f, c_n, Sigma_n, R = R, scalar = scalar, exclude_origin = exclude_origin)
        bias[r]= torch.mean(res* mu)
    return bias.double()

def wishart_deteqv_cov(f_list, N, Sigma_n, R=100, scalar1=1.5, scalar2=2, exclude_origin=False):
    '''
    Numerical evaluation of deterministic equivalent covariance matrix Lambda_n 
    specialized to classical Wishart matrices Bn = (1/N)Sigma_n^{1/2}X_nX_n^TSigma_n^{1/2}.

    Parameters
	----------
    f_list: list of callables
        functions [f_1,...,f_K]
    N: int
        number of columns in X_n
    Sigma_n: torch.Tensor
        population covariance matrix of the Wishart matrix
    R: int
        number of function evaluations in the trapezoidal rule
    exclude_origin: bool
        if the contour excludes the origin 
    scalar1, scalar2: float
        Must be > 1.
        - if exclude_origin = False: radius of contour 1 equal to scalar1 * minimum_possible_radius
                                     radius of contour 2 equal to scalar2 * minimum_possible_radius
        - if exclude_orign = True: radius of contour 1 equal to 1/scalar2 * maximum_possible_radius
                                   radius of contour 2 equal to 1/scalar1 * maximum_possible_radius
    
    Returns 
    ----------
    cov: torch.Tensor
        numerically evaluated covariance matrix Lambda_n
    '''
    K = len(f_list)
    n = Sigma_n.shape[0]
    c_n = n/N
    white  = torch.allclose(Sigma_n, torch.eye(Sigma_n.shape[0], dtype=Sigma_n.dtype, device=Sigma_n.device))

    Sigma_list = [Sigma_n.to(dtype=torch.complex64)]
    Sigmas_tensor = torch.stack(Sigma_list)
    Ls_tensor = torch.stack([(torch.ones(N)).to(dtype=torch.complex64)])

    cov=torch.zeros((K,K),dtype=torch.complex64)
    sigma2=sigma_trap_wishart(Sigmas_tensor,Ls_tensor,R=R,scalar1=scalar1,scalar2=scalar2,exclude_origin=exclude_origin,white=white)
    for r1 in range(K):
        for r2 in range(K):
            f=f_list[r1];g=f_list[r2]
            res=fg_trap_wishart(f,g,c_n,Sigma_n,R=R,scalar1=scalar1,scalar2=scalar2,exclude_origin=exclude_origin)
            cov[r1,r2]= torch.mean(res* sigma2)
    return cov.double()

def wishart_deteqv_lss(f_list,N,Sigma_n,R=100,scalar=2,exclude_origin=False):
    '''
    Numerical evaluation of deterministic equivalent LSS
    specialized to classical Wishart matrices Bn = (1/N)Sigma_n^{1/2}X_nX_n^TSigma_n^{1/2}.

    Parameters
	----------
    f_list: list of callables
        functions [f_1,...,f_K]
    N: int
        number of columns in X_n
    Sigma_n: np.array or torch.Tensor
        population covariance matrix of the Wishart matrix
    R: int
        number of function evaluations in the trapezoidal rule
    exclude_origin: bool
        if the contour excludes the origin 
    scalar: float
        Must be > 1.
        - if exclude_origin = False: radius of contour equal to scalar * minimum_possible_radius
        - if exclude_orign = True: radius of contour equal to 1/scalar * maximum_possible_radius
    
    Returns 
    ----------
    deteqvs: torch.Tensor
        numerically evaluated determinisitc equivalent LSS
    '''
    K = len(f_list)
    n = Sigma_n.shape[0]

    Sigma_list = [Sigma_n.to(dtype=torch.complex64)]
    Sigmas_tensor = torch.stack(Sigma_list)
    Ls_tensor = torch.stack([(torch.ones(N)).to(dtype=torch.complex64)])

    C = n/N
    s_Sigma = torch.linalg.svdvals(Sigmas_tensor[0].double())[0]
    m_Sigma = torch.linalg.svdvals(Sigmas_tensor[0].double())[-1]
    imag = torch.tensor(complex(0,1))
    lam_max=(1+torch.sqrt(torch.tensor(C)))**2**s_Sigma

    if C<1 and exclude_origin:
        lam_min= (1-torch.sqrt(torch.tensor(C)))**2*m_Sigma
        scale = (lam_max-lam_min)/2+lam_min/scalar;
        print(f"function should be analytic on an interval containing [{lam_min},{lam_max}]\nLower limit of the contour is {lam_min*(1-1/scalar)}")
    else:
        lam_min = 0
        scale = (lam_max+lam_min)/2*scalar;
    displacement = (lam_max+lam_min)/2
    
    j = torch.arange(1,R+1)
    z= torch.exp(2*torch.pi*imag*j/R)

    def stieltjes_scaled(z_i):
        zi = scale * z_i + displacement
        m, _ = det_equiv_stieltjes(zi,Sigmas_tensor,Ls_tensor)
        inte =m
        return -inte.detach()*scale
    result = apply_vec(z,stieltjes_scaled,progress="Evaluating determinisitc equivalent LSS...")
    deteqvs=torch.zeros(K)
    for r in range(K):
        f=f_list[r]
        res=f_trap_wishart(f,C,Sigma_n,R=R,scalar=scalar,exclude_origin=exclude_origin)
        deteqvs[r]=torch.mean(res*result)
    return deteqvs

def sig2_wishart(z1,z2,Sigmas_tensor,Ls_tensor,white=False):
    '''
    Evaluation of the function sigma_n^2(z1,z2) given in Lemma 2.3
    specialized to classical Wishart matrices Bn = (1/N)Sigma_n^{1/2}X_nX_n^TSigma_n^{1/2}.
    We verify in Supplementary Appendix G that this implementation 
    agrees with the asymptotic expression in Bai and Silverstein 2004 as n -> infty.

    Parameters
	----------
    white: bool
        if the Wishart matrix is white, in other words, whether Sigma is the identity
        if true, there is a simple closed form expression for the gradient of the Stieltjes transform
    '''
    if not white:
        if not isinstance(z1, torch.Tensor):
            z1 = torch.tensor(z1, dtype=torch.complex64, requires_grad=True)
        else:
            z1.requires_grad_(True)

        if not isinstance(z2, torch.Tensor):
            z2 = torch.tensor(z2, dtype=torch.complex64, requires_grad=True)
        else:
            z2.requires_grad_(True)

    n=Sigmas_tensor.shape[-1]
    N=Ls_tensor.shape[-1]
    c_n = n/N

    m_1_, _ = det_equiv_stieltjes(z1,Sigmas_tensor,Ls_tensor)
    m_2_, _ = det_equiv_stieltjes(z2,Sigmas_tensor,Ls_tensor)
    m_1 = -(1-c_n)/z1+c_n*m_1_
    m_2 = -(1-c_n)/z2+c_n*m_2_
    if white:
        grad_m_1 = -m_1*(1+m_1)/(2*z1*m_1+1-c_n+z1)
        grad_m_2 = -m_2*(1+m_2)/(2*z2*m_2+1-c_n+z2)
    else:
        grad_m1_z1_real = torch.autograd.grad(m_1.real, z1, create_graph=True)[0]
        grad_m_1 = grad_m1_z1_real.conj()
        grad_m2_z2_real = torch.autograd.grad(m_2.real, z2, create_graph=True)[0]
        grad_m_2 = grad_m2_z2_real.conj()
        z1.requires_grad = False
        z2.requires_grad = False
    res = grad_m_1*grad_m_2/(m_1-m_2)**2-1/(z1-z2)**2    
    return res

def mu_wishart(z,Sigmas_tensor,Ls_tensor):
    '''
    Evaluation of the function mu_n(z) given in Lemma 2.3
    specialized to classical Wishart matrices Bn = (1/N)Sigma_n^{1/2}X_nX_n^TSigma_n^{1/2}.
    We verify in Supplementary Appendix G that this implementation 
    agrees with the asymptotic expression in Bai and Silverstein 2004 as n -> infty.
    '''
    n=Sigmas_tensor.shape[-1]
    N=Ls_tensor.shape[-1]
    c_n=n/N
    Sigma = Sigmas_tensor[0]
    m, _ = det_equiv_stieltjes(z,Sigmas_tensor,Ls_tensor)
    m = -(1-c_n)/z+c_n*m
    num = c_n*m**3*torch.trace(torch.pow(Sigma,2)/torch.pow(1+m*Sigma,3))/n
    denom = (1-c_n*m**2*torch.trace(torch.pow(Sigma,2)/torch.pow(1+m*Sigma,2))/n)**2
    res = num/denom
    return res

def sigma_trap_wishart(Sigmas_tensor,Ls_tensor,R=100,scalar1=1.5,scalar2=2,exclude_origin=False,white=False):
    '''
    Evaluation of sigma_n^2(z_1[i],z_2[j]) at each point on the trapezoidal rule grids,
    with z_1 being evaluation points on contour 1 and z_2 on contour 2,
    specialized to classical Wishart matrices Bn = (1/N)Sigma_n^{1/2}X_nX_n^TSigma_n^{1/2}. 

    Parameters
	----------
    white: bool
        if the Wishart matrix is white, in other words, whether Sigma is the identity
        if true, there is a simple closed form expression for the gradient of the Stieltjes transform

    Returns 
    ----------
    res: torch.tensor
        R x R evaluvations of sigma_n^2(z_1[i],z_2[j]) at each point of evaluation for the trapezoidal rule.
    '''
    n=Sigmas_tensor.shape[-1]; N=Ls_tensor.shape[-1]
    C = n/N
    
    s_Sigma = torch.linalg.svdvals(Sigmas_tensor[0].double())[0]
    m_Sigma = torch.linalg.svdvals(Sigmas_tensor[0].double())[-1]
    imag = torch.tensor(complex(0,1))
    lam_max=(1+torch.sqrt(torch.tensor(C)))**2**s_Sigma

    if C<1 and exclude_origin:
        lam_min= (1-torch.sqrt(torch.tensor(C)))**2*m_Sigma
        scale_1 = (lam_max-lam_min)/2+lam_min/scalar2;
        scale_2 = (lam_max-lam_min)/2+lam_min/scalar1;
        print(f"function should be analytic on an interval containing [{lam_min},{lam_max}]\nLower limit of contour one is {lam_min*(1-1/scalar2)}, Lower limit of contour two is {lam_min*(1-1/scalar1)}")
    else:
        lam_min = 0
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
        inte =sig2_wishart(z1i,z2j,Sigmas_tensor,Ls_tensor,white=white)
        return inte.detach()
    result = apply_grid(z_1,z_2,sigma2_scaled,progress="Evaluating covariance...")
    return result*2*scale_1*scale_2

def fg_trap_wishart(f,g,c_n,Sigma_n,R=100,scalar1=1.5,scalar2=2, exclude_origin=False):
    '''
    Evaluation of f(z_1[i])g(z_2[j]) at each point on the trapezoidal rule grids,
    with z_1 being evaluation points on contour 1 and z_2 on contour 2,
    specialized to classical Wishart matrices Bn = (1/N)Sigma_n^{1/2}X_nX_n^TSigma_n^{1/2}. 

    Returns 
    ----------
    res: torch.tensor
        R x R evaluvations of f(z_1[i])g(z_2[j]) at each point of evaluation for the trapezoidal rule.
    '''
    s_Sigma = torch.linalg.svdvals(Sigma_n.double())[0]
    m_Sigma = torch.linalg.svdvals(Sigma_n.double())[-1]

    imag = torch.tensor(complex(0,1))
    lam_max=(1+torch.sqrt(torch.tensor(c_n)))**2*s_Sigma
    if c_n<1 and exclude_origin:
        lam_min= (1-torch.sqrt(torch.tensor(c_n)))**2*m_Sigma
        scale_1 = (lam_max-lam_min)/2+lam_min/scalar2;
        scale_2 = (lam_max-lam_min)/2+lam_min/scalar1;
    else:
        lam_min = 0
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

def mu_trap_wishart(Sigmas_tensor,Ls_tensor,R=100,scalar=2,exclude_origin=False):
    '''
    Evaluation of mu_n(z[i]) at each contour point used in the trapezoidal rule,
    specialized to classical Wishart matrices Bn = (1/N)Sigma_n^{1/2}X_nX_n^TSigma_n^{1/2}. 

    Returns 
    ----------
    res: torch.tensor
        R evaluvations of mu_n^2(z[i]) at each point of evaluation used in the trapezoidal rule.
    '''
    n=Sigmas_tensor.shape[-1]; N=Ls_tensor.shape[-1]
    C = n/N
    
    s_Sigma = torch.linalg.svdvals(Sigmas_tensor[0].double())[0]
    m_Sigma = torch.linalg.svdvals(Sigmas_tensor[0].double())[-1]
    imag = torch.tensor(complex(0,1))
    lam_max=(1+torch.sqrt(torch.tensor(C)))**2**s_Sigma

    if C<1 and exclude_origin:
        lam_min= (1-torch.sqrt(torch.tensor(C)))**2*m_Sigma
        scale = (lam_max-lam_min)/2+lam_min/scalar;
        print(f"function should be analytic on an interval containing [{lam_min},{lam_max}]\nLower limit of the contour is {lam_min*(1-1/scalar)}")
    else:
        lam_min = 0
        scale = (lam_max+lam_min)/2*scalar;

    displacement = (lam_max+lam_min)/2
    
    j = torch.arange(1,R+1)
    z= torch.exp(2*torch.pi*imag*j/R)
    def mu_scaled(z_i):
        zi = scale * z_i + displacement
        inte =mu_wishart(zi,Sigmas_tensor,Ls_tensor)
        return inte.detach()
    result = apply_vec(z,mu_scaled,progress="Evaluating bias...")
    return -result*scale

def f_trap_wishart(f,c_n,Sigma_n,R=100,scalar=2,exclude_origin=False):
    '''
    Evaluation of f(z[i]) at each contour point used in the trapezoidal rule,
    specialized to classical Wishart matrices Bn = (1/N)Sigma_n^{1/2}X_nX_n^TSigma_n^{1/2}. 

    Returns 
    ----------
    res: torch.tensor
        R evaluvations of f(z[i]) at each point of evaluation for the trapezoidal rule.
    '''
    s_Sigma = torch.linalg.svdvals(Sigma_n.double())[0]
    m_Sigma = torch.linalg.svdvals(Sigma_n.double())[-1]

    imag = torch.tensor(complex(0,1))
    lam_max=(1+torch.sqrt(torch.tensor(c_n)))**2*s_Sigma
    if c_n<1 and exclude_origin:
        lam_min= (1-torch.sqrt(torch.tensor(c_n)))**2*m_Sigma
        scale = (lam_max-lam_min)/2+lam_min/scalar;
    else:
        lam_min = 0
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
    
