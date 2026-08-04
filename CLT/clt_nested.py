import torch
import numpy as np
from CLT.trapezoidal_cov import main_deteqv_cov
from CLT.trapezoidal_bias import main_deteqv_bias,main_deteqv_lss

def clt_nested(theta_list, U_list, B_list, p, jacobian_F_full, R_lss = 100, R_bias = 100, R_cov = 20):
    '''
    Numerical evaluation of the deterministic equivalent bias and covariance 
    in the CLT for method of moments estimators of the population parameters 
    that parametrize the eigenvalue distribution of the population covariance matrices Sigma_A and Sigma_E
    in the linear mixed effects model constructed under the full sibling design.

    Parameters
	----------
    theta_list: list of lists
        theta_list = [[tau_a,pi],[tau_e]] is a list of the population parameters
    U_list : list of numpy.ndarray
        Membership matrices [U, I_n], where U assigns each individual to the corresponding family.
    B_list : list of numpy.ndarray
        Matrices [B_1, B_2] used to construct sum-of-squares matrices S_1 and S_2
    p: int
        number of traits measured for each individual
    jacobian_F_full: torch.Tensor
        3 x 3 Jacobian matrix of the mapping F from the method of moments estimators to the empirical moments of S_1 and S_2
    R_lss, R_bias, R_cov: int
        number of function evaluations used in the trapezoidal rule when numerically evaluating the deterministic lss, bias and covariance


    Returns 
    ----------
    bias_theta: torch.Tensor
        numerically evaluated bias vector of length 3
    cov_theta: torch.Tensor
        numerically evaluated covariance matrix of size 3 x 3
    '''
    k=len(theta_list)
    n = B_list[0].shape[0];I=U_list[0].shape[1]
    K_list = [len(x) for x in theta_list]
    (tau_a,pi),tau_e=theta_list
    SigmaA_diag = torch.tensor([tau_a]*int(p*pi)+[0]*(p-int(p*pi)))
    SigmaE_diag = torch.tensor([tau_e]*p)
    Sigma_diag_list = [SigmaA_diag,SigmaE_diag]

    Bias_mmts_full = torch.zeros(sum(K_list))
    Det_alpha_full = torch.zeros(sum(K_list))
    Cov_mmts_full = torch.zeros((sum(K_list),sum(K_list)))       

    for index in range(k-1,-1,-1):
        print(f'#### Level {index+1} ####')
        Gamma_list = [torch.tensor(B_list[index]@U_list[index+r]@U_list[index+r].T@B_list[index]) for r in range(k-index)]
        L_diag_list = Sigma_diag_list[-(k-index):]
        f_list = [lambda x, r=r: x**r for r in range(1,K_list[index]+1)]
        cov = main_deteqv_cov(f_list,Gamma_list,L_diag_list,R=R_cov)
        if index==k-1:
            Cov_mmts_full[-sum(K_list[index:]):,-sum(K_list[index:]):] = cov
        else:
            Cov_mmts_full[-sum(K_list[index:]):-sum(K_list[index+1:]),-sum(K_list[index:]):-sum(K_list[index+1:])] = cov
        inv_jac = torch.inverse(jacobian_F_full[-sum(K_list[index:]):,-sum(K_list[index:]):])
        cov_theta=torch.matmul(torch.matmul(inv_jac,Cov_mmts_full[-sum(K_list[index:]):,-sum(K_list[index:]):]),
                                        torch.transpose(inv_jac,0,1))/(p**2)
        bias = main_deteqv_bias(f_list,Gamma_list,L_diag_list,R=R_bias)
        if index==k-1:
            Bias_mmts_full[-sum(K_list[index:]):] = bias
        else:
            Bias_mmts_full[-sum(K_list[index:]):-sum(K_list[index+1:])] = bias
        bias_theta=torch.matmul(torch.inverse(jacobian_F_full[-sum(K_list[index:]):,-sum(K_list[index:]):]),
                                    Bias_mmts_full[-sum(K_list[index:]):])/p
        det_alpha = main_deteqv_lss(f_list,Gamma_list,L_diag_list,R=R_lss)*n/p
        if index==k-1:
            Det_alpha_full[-sum(K_list[index:]):] = det_alpha
        else:
            Det_alpha_full[-sum(K_list[index:]):-sum(K_list[index+1:])] = det_alpha

    tau_e_hat = Det_alpha_full[2]*p/(n-I)
    F1 = B_list[0] @ U_list[0] @ U_list[0].T; F2 = B_list[0] @ U_list[1] @ U_list[1].T
    term1 = (Det_alpha_full[1] - (np.trace(F2@F2)/p + (np.trace(F2)/p)**2)*tau_e_hat**2)/(np.trace(F1)/p)/(Det_alpha_full[0]-np.trace(F2)/p*tau_e_hat)
    term2 = (np.trace(F1@F1)/p)*(Det_alpha_full[0]-np.trace(F2)/p*tau_e_hat)/(np.trace(F1)/p)**3
    term3 = 2*tau_e_hat*(1+np.trace(F2)/p)/(np.trace(F1)/p)
    tau_a_hat = term1 - term2 - term3
    pi_hat = (Det_alpha_full[0]-np.trace(F2)/p*tau_e_hat)/(np.trace(F1)/p)/tau_a_hat

    Bias_det_full = torch.tensor([tau_a_hat-tau_a,pi_hat-pi,tau_e_hat-tau_e])
    bias_theta += Bias_det_full
    return bias_theta.detach(), cov_theta


