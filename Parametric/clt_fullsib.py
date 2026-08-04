import torch
import os,sys
import warnings
import math
import numpy as np
from scipy import optimize

sys.path.insert(0, os.path.abspath(".."))
from CLT.trapezoidal_cov import main_deteqv_cov
from CLT.trapezoidal_bias import main_deteqv_bias,main_deteqv_lss
from Parametric.utils_model_design import generate_owub_design

pi0=0.8

def avg_trace_s5(pi_hat, eta=torch.tensor(torch.pi/4)):
    t1 = torch.cos(eta)**2
    t2 = torch.sin(eta)**2
    if pi_hat<pi0:
        if pi_hat<1-pi0:
            return t1*pi_hat
        return t1*pi_hat+t2*(pi_hat+pi0-1)
    return t1*pi0+t2*(pi_hat+pi0-1)

def Fn_joint(params,F_trace,scenario):
    tau1, pi, tau2 = params 
    if scenario == '2' or scenario== '3':
        avg_trace=pi
    if scenario == '5':
        avg_trace = avg_trace_s5(pi)

    x = [pi*tau1,pi0*tau2, 
         pi*tau1**2, tau1*tau2*avg_trace,pi0*tau2**2,
         pi*tau1**3, tau1**2*tau2*avg_trace, 
         tau1*tau2**2*avg_trace,pi0*tau2**3]
    F1_1,F2_1,F1_2,F1F2,F2_2,F1_3,F1_2_F2,F1_F2_2,F2_3=F_trace
    m1 = F1_1*x[0] + F2_1*x[1]
    m2 = (F1_2*x[0]**2 + 2*F1F2*x[0]*x[1]+ F2_2*x[1]**2 
          + F1_1**2*x[2] + 2*F1_1*F2_1*x[3] + F2_1**2*x[4])
    m3 = (F1_3*x[0]**3 + 3*F1_2_F2*x[0]**2*x[1] + 3*F1_F2_2*x[0]*x[1]**2+ F2_3*x[1]**3
          + 3*F1_1*F1_2*x[0]*x[2]+ 3*F1_2*F2_1*x[0]*x[3]+ 3*F1_1*F1F2*x[0]*x[3]+ 3*F2_1*F1F2*x[0]*x[4]
          + 3*F1_1*F1F2*x[1]*x[2]+ 3*F1_1*F2_2*x[1]*x[3]+ 3*F2_1*F1F2*x[1]*x[3]+ 3*F2_1*F2_2*x[1]*x[4]
          + F1_1**3*x[5] + 3*F1_1**2*F2_1*x[6]+ 3*F1_1*F2_1**2*x[7] + F2_1**3*x[8])
    return torch.stack([m1, m2, m3])

def ratio_transform(a):
    out = np.empty_like(a)
    out[0] = a[0]
    out[1:] = a[1:] / a[:-1]
    return out

def Fn_joint_inverse(mu_hat,F_trace,scenario,thresh=0.01):
    mu_hat_transformed = np.array(mu_hat.clone())
    mu_hat_transformed[1:]=mu_hat_transformed[1:]/mu_hat_transformed[:-1]
    theta_hat = np.array([1.,1.,1.])
    Fn_transformed = lambda theta: ratio_transform(Fn_joint(theta, F_trace=F_trace, scenario=scenario))
    while (np.linalg.norm(Fn_transformed(theta_hat) - mu_hat_transformed) > thresh) or (np.any(theta_hat < -0.5)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            theta_hat = optimize.fsolve(lambda  x: Fn_transformed(x) - mu_hat_transformed,
                                        np.random.uniform(0, 1, size=len(mu_hat_transformed)),maxfev=5000)
    return theta_hat

def Q(p, eta):
    eta = torch.tensor(eta)
    return torch.cos(eta)*torch.eye(p,dtype=torch.double)+torch.sin(eta)*torch.fliplr(torch.diag(torch.tensor([-1]*int(p/2)+[1]*int(p/2),dtype=torch.double)))

def clt_joint_fullsib(theta_true, B, U_list, J_list, p, scenario, R_cov = 20, R_bias = 100, R_lss = 100, det_thresh = 0.001, factor=10):
    '''
    Scenarios 2,3,5
    '''
    tau1, pi, tau2 = theta_true 
    I= U_list[0].shape[1]; p_small = int(p/factor); I_small = int(I/factor)

    B = torch.tensor(B); U_list = [torch.tensor(U) for U in U_list]
    F1 = B @ U_list[0] @ U_list[0].T; F2 = B @ U_list[1] @ U_list[1].T
    F_trace = [torch.trace(F1)/p,torch.trace(F2)/p,
        torch.trace(F1@F1)/p,torch.trace(F1@F2)/p,torch.trace(F2@F2)/p,
        torch.trace(F1@F1@F1)/p,torch.trace(F1@F1@F2)/p,torch.trace(F1@F2@F2)/p,torch.trace(F2@F2@F2)/p]

    if scenario == '2': 
        mat_eigvec1 = torch.eye(p,dtype=torch.double)
        mat_eigvec1_small = torch.eye(p_small,dtype=torch.double)
    elif scenario == '3':
        mat_eigvec1 = Q(p, eta=torch.pi/4)
        mat_eigvec1_small = Q(p_small, eta=torch.pi/4)
    elif scenario == '5':
        mat_eigvec1 = Q(p, eta=torch.pi/4)
        mat_eigvec1_small = Q(p_small, eta=torch.pi/4)
    else:
        print("ERROR: Undefined scenario.")
        return
    
    mat_eigvec2_small = torch.eye(p_small,dtype=torch.double)
    mat_eigvec2 = torch.eye(p,dtype=torch.double)

    U_list_small, _, _, _ = generate_owub_design(I_small, J_list=J_list[:I_small], include_intercept=False)
    U_list_small = [torch.tensor(U) for U in U_list_small]
    
    Sigma1_eig_small = torch.tensor([tau1]*int(p_small*pi)+[0]*(p_small-int(p_small*pi)), dtype= torch.double)
    Sigma2_eig_small = torch.tensor([tau2]*int(p_small*pi0)+[0]*(p_small-int(p_small*pi0)), dtype= torch.double)
    Sigma_list_small = [mat_eigvec1_small@torch.diag(Sigma1_eig_small)@mat_eigvec1_small.T,
                        mat_eigvec2_small@torch.diag(Sigma2_eig_small)@mat_eigvec2_small.T]
    L_diag_list_small = [torch.diag(U_list_small[0].T@U_list_small[0]),torch.ones(I_small)]
    
    f_list = [lambda x, r=r: x**r for r in range(1,len(theta_true)+1)]

    cov = main_deteqv_cov(f_list,Sigma_list_small,L_diag_list_small,R=R_cov)
    for i in range(len(theta_true)):
        for j in range(len(theta_true)):
            cov[i,j]=cov[i,j]*(I/p)**(i+j+2)

    bias =  main_deteqv_bias(f_list,Sigma_list_small,L_diag_list_small,R=R_bias) 
    for i in range(len(theta_true)):
        bias[i]=bias[i]*(I/p)**(i+1)
        
    Sigma1_eig = torch.tensor([tau1]*int(p*pi)+[0]*(p-int(p*pi)), dtype= torch.double)
    Sigma2_eig = torch.tensor([tau2]*int(p*pi0)+[0]*(p-int(p*pi0)), dtype= torch.double)
    Sigma_list = [mat_eigvec1@torch.diag(Sigma1_eig)@mat_eigvec1.T,
                mat_eigvec2@torch.diag(Sigma2_eig)@mat_eigvec2.T]
    L_diag_list = [torch.diag(U_list[0].T@U_list[0]),torch.ones(I)]

    f_list = [lambda x, r=r: x**r for r in range(1,len(theta_true)+1)]
    det_alpha = main_deteqv_lss(f_list,Sigma_list,L_diag_list,R=R_lss)
    for i in range(len(theta_true)):
        det_alpha[i]=det_alpha[i]*(I/p)**(i+1)
    det_theta = Fn_joint_inverse(det_alpha,F_trace, scenario, thresh=det_thresh)
    det_bias = torch.tensor(det_theta)-torch.tensor(theta_true)

    params = torch.tensor(det_theta, dtype=torch.double, requires_grad=True)
    Jacobian_Fn =  torch.autograd.functional.jacobian(lambda x: Fn_joint(x,F_trace=F_trace, scenario=scenario),params)
    inv_jac = torch.inverse(Jacobian_Fn)
    bias_theory=torch.matmul(inv_jac,bias)/p+det_bias
    cov_theory=torch.matmul(torch.matmul(inv_jac,cov),torch.transpose(inv_jac,0,1))/(p**2)
    #sd_theory = np.sqrt(np.diag(cov_theory))
    
    return bias_theory, cov_theory

def Fn_seq(params,B_list,U_list,p):
    tau1, pi, tau3 = params
    n = U_list[0].shape[0]; I= U_list[0].shape[1]
    F1 = torch.tensor(B_list[0] @ U_list[0] @ U_list[0].T); F2 = torch.tensor(B_list[0] @ U_list[1] @ U_list[1].T)
    x = [pi*tau1,pi0*tau3, pi*tau1**2, tau1*tau3*torch.min(pi,torch.tensor(pi0)),pi0*tau3**2]
    m1 = (torch.trace(F1)/p)*x[0] + torch.trace(F2)/p*x[1]
    m2 = ((torch.trace(F1@F1)/p)*x[0]**2 + 2 * (torch.trace(F1@F2)/p)*x[0]*x[1]+ (torch.trace(F2@F2)/p)*x[1]**2 
          + (torch.trace(F1)/p)**2*x[2] + 2*(torch.trace(F1)/p)*(torch.trace(F2)/p)*x[3] + (torch.trace(F2)/p)**2*x[4])
    m3 = (n-I)/p*pi0*tau3
    return torch.stack([m1, m2, m3])

def Fn_seq_inv(mu_hat,F_trace,p,n,I):
    '''
    mu = [mu_1^{(1)},mu_1^{(2)},mu_2]
    '''
    mu1_1,mu1_2,mu2 = mu_hat
    tau_2_hat = p/(n-I) * mu2 /pi0
    

    F1_1,F2_1,F1_2,_,F2_2,_,_,_,_=F_trace
    term1 = (mu1_2 - (F2_2*(tau_2_hat*pi0)**2 + F2_1**2*tau_2_hat**2*pi0))/F1_1/(mu1_1-F2_1*pi0*tau_2_hat)
    term2 = F1_2*(mu1_1-F2_1*pi0*tau_2_hat)/F1_1**3
    term3 = 2*pi0*tau_2_hat/F1_1
    A = term1 - term2 - term3 
    b =  -2*F2_1/F1_1*tau_2_hat
    c = F1_1*pi0/(mu1_1-F2_1*pi0*tau_2_hat)
    if (A+b)*c >= 1:
        tau_1_hat = A+b
    else:
        tau_1_hat = A/(1-b*c)
    pi_hat = (mu1_1-F2_1*pi0*tau_2_hat)/F1_1/tau_1_hat

    theta_hat_seq = np.array([tau_1_hat,pi_hat,tau_2_hat])  
    return theta_hat_seq

def clt_seq_fullsib(theta_true,B_list,U_list,J_list,p,R_cov = 20,R_bias = 100, R_lss=100, factor = None):
    '''
    Scenario 1
    '''
    k=2
    n = U_list[0].shape[0]; I=U_list[0].shape[1]
    K_list = [2,1]

    tau1, pi, tau2 = theta_true

    params = torch.tensor(theta_true, dtype=torch.double, requires_grad=True)
    jacobian_F_full = torch.autograd.functional.jacobian(lambda x: Fn_seq(x,B_list,U_list,p),params)
    
    if factor is not None:
        p_small = int(p/factor); I_small = int(I/factor)
        U_list_small, B_list_small, _, n_small = generate_owub_design(int(I_small), J_list=J_list[:int(I_small)], include_intercept=False)
    else:
        p_small = p; I_small = I; n_small = n
        U_list_small, B_list_small = U_list,B_list
    Sigma1_eig_small = torch.tensor([tau1]*int(p_small*pi)+[0]*(p_small-int(p_small*pi)))
    Sigma2_eig_small = torch.tensor([tau2]*int(p_small*pi0)+[0]*(p_small-int(p_small*pi0)))
    Sigma_diag_list_small = [Sigma1_eig_small,Sigma2_eig_small]

    Bias_mmts_full = torch.zeros(sum(K_list), dtype=torch.double)
    Det_mu_full = torch.zeros(sum(K_list), dtype=torch.double)
    Cov_mmts_full = torch.zeros((sum(K_list),sum(K_list)), dtype=torch.double)       

    for index in range(k-1,-1,-1):
        print(f'#### Level {index+1} ####')
        Sigma_list = [torch.tensor(B_list_small[index]@U_list_small[index+r]@U_list_small[index+r].T@B_list_small[index]) for r in range(k-index)]
        L_diag_list = Sigma_diag_list_small[-(k-index):]

        f_list = [lambda x, r=r: x**r for r in range(1,K_list[index]+1)]
        cov = main_deteqv_cov(f_list,Sigma_list,L_diag_list,R=R_cov)
        if index==k-1:
            Cov_mmts_full[-sum(K_list[index:]):,-sum(K_list[index:]):] = cov
        else:
            Cov_mmts_full[-sum(K_list[index:]):-sum(K_list[index+1:]),-sum(K_list[index:]):-sum(K_list[index+1:])] = cov
        inv_jac = torch.inverse(jacobian_F_full[-sum(K_list[index:]):,-sum(K_list[index:]):])
        cov_theta=torch.matmul(torch.matmul(inv_jac,Cov_mmts_full[-sum(K_list[index:]):,-sum(K_list[index:]):]),
                                        torch.transpose(inv_jac,0,1))/(p**2)

        bias = main_deteqv_bias(f_list,Sigma_list,L_diag_list,R=R_bias)
        if index==k-1:
            Bias_mmts_full[-sum(K_list[index:]):] = bias
        else:
            Bias_mmts_full[-sum(K_list[index:]):-sum(K_list[index+1:])] = bias
        bias_theta=torch.matmul(torch.inverse(jacobian_F_full[-sum(K_list[index:]):,-sum(K_list[index:]):]),
                                    Bias_mmts_full[-sum(K_list[index:]):])/p
        
        det_alpha = main_deteqv_lss(f_list,Sigma_list,L_diag_list,R=R_lss)*n_small/p_small
        
        if index==k-1:
            Det_mu_full[-sum(K_list[index:]):] = det_alpha
        else:
            Det_mu_full[-sum(K_list[index:]):-sum(K_list[index+1:])] = det_alpha

    F1 = torch.tensor(B_list_small[0] @ U_list_small[0] @ U_list_small[0].T); F2 = torch.tensor(B_list_small[0] @ U_list_small[1] @ U_list_small[1].T)
    F_trace = [torch.trace(F1)/p_small,torch.trace(F2)/p_small,
            torch.trace(F1@F1)/p_small,torch.trace(F1@F2)/p_small,torch.trace(F2@F2)/p_small,
            torch.trace(F1@F1@F1)/p_small,torch.trace(F1@F1@F2)/p_small,torch.trace(F1@F2@F2)/p_small,torch.trace(F2@F2@F2)/p_small]
    det_theta = Fn_seq_inv(Det_mu_full,F_trace,p_small,n_small,I_small) 
    det_bias = det_theta-theta_true
    #sd_theta = np.sqrt(np.diag(cov_theta))
    return bias_theta.detach()+det_bias, cov_theta 


