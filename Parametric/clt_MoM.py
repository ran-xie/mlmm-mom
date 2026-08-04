import torch
import sys, os
import numpy as np
from Parametric.Phi import Phi,theta_to_Sig_diag
from Parametric.Fn import jacobian_Fn_joint,Fn_inverse_joint_deteqv
from Parametric.Fn_nested import Fn_inverse_nested_deteqv,jacobian_Fn_nested

sys.path.insert(0, os.path.abspath(".."))
from CLT.trapezoidal_cov import main_deteqv_cov
from CLT.trapezoidal_bias import main_deteqv_bias,main_deteqv_lss

def clt_joint(theta_list,B,U_list,decay_list,p,R_cov = 20,R_bias = 100, R_lss=100,det_thresh=0.001):
    k=len(theta_list)
    n = B.shape[0]
    K_list = [len(x) for x in theta_list]
    K = sum(K_list)
    Sigma_diag_list = [theta_to_Sig_diag(theta_list[i],p,decay=decay_list[i]) for i in range(k)]
    jacobian_F_full = jacobian_Fn_joint(theta_list,B,U_list,decay_list,p)
    
    Gamma_list = [torch.tensor(B@U_list[r]@U_list[r].T@B) for r in range(k)]
    L_diag_list = Sigma_diag_list

    f_list = [lambda x, r=r: x**r for r in range(1,K+1)]
    cov = main_deteqv_cov(f_list,Gamma_list,L_diag_list,R=R_cov)
    inv_jac = torch.inverse(jacobian_F_full).double()
    cov_theta=torch.matmul(torch.matmul(inv_jac,cov),torch.transpose(inv_jac,0,1))/(p**2)

    bias = main_deteqv_bias(f_list,Gamma_list,L_diag_list,R=R_bias)
    bias_theta=torch.matmul(inv_jac,bias)/p
    
    det_alpha = main_deteqv_lss(f_list,Gamma_list,L_diag_list,R=R_lss)*n/p
    with torch.no_grad():
        det_theta_full = Fn_inverse_joint_deteqv(det_alpha,B,U_list,decay_list,K_list,p,thresh=det_thresh)
    Bias_det_full = np.concatenate(det_theta_full)-np.concatenate(theta_list)
    return bias_theta.detach()+Bias_det_full, cov_theta.detach() 




def clt_nested(theta_list,B_list,U_list,decay_list,p,R_cov = 20,R_bias = 100, R_lss=1000,det_thresh=0.0001):
    k=len(theta_list)
    n = B_list[0].shape[0]
    K_list = [len(x) for x in theta_list]
    Sigma_diag_list = [theta_to_Sig_diag(theta_list[i],p,decay=decay_list[i]) for i in range(k)]
    jacobian_F_full = jacobian_Fn_nested(theta_list,B_list,U_list,decay_list,p,verbose=False)

    Bias_mmts_full = torch.zeros(sum(K_list))
    Cov_mmts_full = torch.zeros((sum(K_list),sum(K_list)))       
    det_alpha_list = [None]*k

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
        bias_theta=torch.matmul(inv_jac,Bias_mmts_full[-sum(K_list[index:]):])/p
        
        det_alpha = main_deteqv_lss(f_list,Gamma_list,L_diag_list,R=R_lss)*n/p
        det_alpha_list[index] = det_alpha
    det_theta_full = Fn_inverse_nested_deteqv(det_alpha_list,B_list,U_list,decay_list,p,thresh=det_thresh)
    Bias_det_full = np.concatenate(det_theta_full)-np.concatenate(theta_list)
    return bias_theta.detach()+Bias_det_full, cov_theta.detach()


