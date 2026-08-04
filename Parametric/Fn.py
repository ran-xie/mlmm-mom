import os
import torch
import numpy as np
from scipy import optimize
from itertools import accumulate
from numpy.linalg import matrix_power


from Parametric.Phi import Phi
from Parametric.Psi_n import clear_cache, Psi_n


cache_folder_default = f"cache_find_theta/cache_find_theta_fraction"

def Fn_inverse_joint(Y, B, U_list, decay_list, K_list, thresh=0.005, clamp_max=20.0, clamp_min=-1.0):
    """
    Solves for method of moments estimators of theta jointly from a single Sn, 
    following the procedure described in Section 2.1.

    Parameters
    ----------
    Y: n x p observations
    """
    n,p = Y.shape; k=len(decay_list)
    theta_hat_list = [None] * k
    K = sum(K_list)

    S = (Y.T @ B @ Y) / p
    trace_moments = np.array([0.0] * K)
    for i in range(K):
        trace_moments[i] = np.trace(matrix_power(S, i + 1)) / p
    theta_to_sigmom = (
        lambda x: Phi([x[sum(K_list[:index]):sum(K_list[:(index+1)])] for index in range(k)], decay_list, K, p)
    )
    theta_hat, _ = Fn_inverse(
        trace_moments, p, n, B, U_list,
        theta_to_sigmom,
        thresh=thresh, clamp_max=clamp_max, clamp_min=clamp_min
    )
    for index in range(k):
        theta_hat_index = theta_hat[sum(K_list[:index]):sum(K_list[:index+1])]
        if decay_list[index]=='step':
            taus = theta_hat_index[:int((K_list[index]+1)/2)]
            pis = np.append(theta_hat_index[int((K_list[index]+1)/2):],1-sum(theta_hat_index[int((K_list[index]+1)/2):]))
            idx_sorted = np.argsort(taus)[::-1]
            theta_hat_index = np.concatenate((taus[idx_sorted],pis[idx_sorted][:-1]))
        theta_hat_list[index] = theta_hat_index
    return theta_hat_list


def Fn_inverse(mmts, p, n, B, U_list, theta_to_m, thresh=0.005, clamp_max=20.0, clamp_min=-1.0, cache_folder = cache_folder_default, seed = None):
    """
    Solves for method of moments estimators of theta from moments of Sn.

    Parameters
    ----------
    mmts: vector of moments of Sn
    theta_to_m:  a function that takes theta to mixed population moments
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    clear_cache()
    L=len(mmts)
    mmts = mmts.copy()
    mmts[1:] = mmts[1:] / mmts[:-1] 
    theta_hat = np.array([0.]*len(mmts))
    F_check_list = []
    for r in range(len(U_list)):
        F_check_list.append(B@U_list[r]@U_list[r].T)

    def Fn(theta):
        theta = torch.tensor(theta)
        theta = torch.clamp(theta, min=clamp_min, max=clamp_max)
        if torch.isnan(theta).any():
            print("NaN detected in theta before computation:", theta)
        
        key_to_pos_dict,Sig_mom_list = theta_to_m(theta)
        moments = Psi_n(Sig_mom_list, L, p, n, B, U_list, key_to_pos_dict,from_find_theta=True,F_check_list=F_check_list,cache_folder=cache_folder)
        clamped_denominator = torch.where(
            moments[:-1] > 0, 
            torch.clamp(moments[:-1], min=1e-3),  
            torch.clamp(moments[:-1], max=-1e-3)  
        )
        moments[1:] = moments[1:] / clamped_denominator
        return moments.detach().numpy() 
    
    def solve_equation():
        theta_hat = optimize.fsolve(lambda  x: Fn(x) - mmts,[np.random.uniform(0,1,1)]*len(mmts),maxfev=5000)#
        return theta_hat
    
    while (np.linalg.norm(Fn(theta_hat) - mmts) > thresh) or (np.any(theta_hat > clamp_max)) or (np.any(theta_hat < clamp_min)):
        theta_hat = solve_equation()

    error = np.linalg.norm(Fn(theta_hat) - mmts)
    return theta_hat, error
    

def x_to_list(x,K_list):
    res = []
    K_list = [0]+K_list
    K_list = list(accumulate(K_list))
    for i in range(len(K_list)-1):
        res.append(x[K_list[i]:K_list[i+1]])
    return res
        

def Fn_inverse_joint_deteqv(trace_moments,B,U_list,decay_list,K_list,p,thresh=0.005,clamp_max=20.0,clamp_min=-1.0):
    n = B.shape[0]; k=len(decay_list)
    theta_hat_list = [None] * k
    K = sum(K_list)
    trace_moments = np.array(trace_moments)

    theta_to_sigmom = (
        lambda x: Phi([x[sum(K_list[:index]):sum(K_list[:(index+1)])] for index in range(k)], decay_list, K, p)
    )
    theta_hat, _ = Fn_inverse(
        trace_moments, p, n, B, U_list,
        theta_to_sigmom,
        thresh=thresh, clamp_max=clamp_max, clamp_min=clamp_min
    )
    for index in range(k):
        theta_hat_index = theta_hat[sum(K_list[:index]):sum(K_list[:index+1])]
        if decay_list[index]=='step':
            taus = theta_hat_index[:int((K_list[index]+1)/2)]
            pis = np.append(theta_hat_index[int((K_list[index]+1)/2):],1-sum(theta_hat_index[int((K_list[index]+1)/2):]))
            idx_sorted = np.argsort(taus)[::-1]
            theta_hat_index = np.concatenate((taus[idx_sorted],pis[idx_sorted][:-1]))
        theta_hat_list[index] = theta_hat_index
    return theta_hat_list

def jacobian_Fn_joint(theta_list,B,U_list,decay_list,p,cache=None):
    theta_full = torch.cat(theta_list)
    n=B.shape[0]

    k = len(theta_list)
    K_list = []
    for r in range(k):
        K_list.append(len(theta_list[r]))
    K=sum(K_list)
    
    if cache is not None:
        cache_folder_jac = 'cache/'+cache+'/'
        if not os.path.exists(cache_folder_jac):
            os.makedirs(cache_folder_jac)
    
    theta_to_sigmom = (
        lambda x: Phi([x[sum(K_list[:index]):sum(K_list[:(index+1)])] for index in range(k)], decay_list, K, p,jac=True)
    )
 
    true_moments = []

    if cache is not None:
        jacobian_Phi_path = os.path.join(cache_folder_jac, f"G_matrix_level_jacobian_Phi.pt")
        moments_path = os.path.join(cache_folder_jac, f"G_matrix_level_moments.pt")
        jacobian_Psi_path = os.path.join(cache_folder_jac, f"G_matrix_level_jacobian_Psi.pt")
        jacobian_F_path = os.path.join(cache_folder_jac, f"G_matrix_level_jacobian_F.pt")
        

    if cache is not None and all(os.path.exists(path) for path in [jacobian_Phi_path, moments_path, jacobian_Psi_path, jacobian_F_path]):
        # Load from cache
        print(f"Loading cached results")
        jacobian_Phi = torch.load(jacobian_Phi_path)
        moments = torch.load(moments_path)
        true_moments.insert(0, moments.tolist())
        print("moments: ",moments.tolist())
        jacobian_Psi = torch.load(jacobian_Psi_path)
        jacobian_F = torch.load(jacobian_F_path)
    else:
        key_to_pos_dict,Sig_mom_list,jacobian_Phi = theta_to_sigmom(theta_full)
        print("jacobian_Phi: ",jacobian_Phi)

        Sig_mom_tensor = Sig_mom_list.clone().detach().requires_grad_(True)
        moments = Psi_n(Sig_mom_tensor, K, p, n, B, U_list, key_to_pos_dict)
        print("moments: ",moments.tolist())

        jacobian_Psi = torch.autograd.functional.jacobian(lambda x: Psi_n(x, K, p, n, B, U_list, key_to_pos_dict),Sig_mom_tensor)
        print("jacobian_Psi: ",jacobian_Psi)
        jacobian_F = jacobian_Psi@jacobian_Phi
        if cache is not None:
            torch.save(jacobian_Psi, os.path.join(cache_folder_jac, f"G_matrix_level_jacobian_Psi.pt"))
            torch.save(jacobian_F, os.path.join(cache_folder_jac, f"G_matrix_level_jacobian_F.pt"))
        
        print(f"jacobian_F: ",jacobian_F)
        
        
        print("#"*60)

    return jacobian_F
