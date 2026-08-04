import os
import torch
import numpy as np
from scipy import optimize
from itertools import accumulate
from numpy.linalg import matrix_power
from Parametric.Fn import Fn_inverse,x_to_list

from Parametric.Phi import Phi
from Parametric.Psi_n import clear_cache, Psi_n


def Fn_inverse_nested(Y,B_list,U_list,decay_list,K_list,thresh=0.005,clamp_max=20.0,clamp_min=-1.0,seed=None):
    """
    Solves for method of moments estimators of theta_r (corresponding to Sigma_r) sequentially
    leveraging the nested structure of the model,
    following the procedure described in Section 2.3.

    Parameters
    ----------
    Y: n x p observations
    """
    n,p = Y.shape; k=len(decay_list)
    theta_hat_list = [None] * k

    for index in range(k - 1, -1, -1):
        S = (Y.T @ B_list[index] @ Y) / p
        trace_moments = np.array([0.0] * K_list[index])
        for i in range(K_list[index]):
            trace_moments[i] = np.trace(matrix_power(S, i + 1)) / p
        solved_tail = [torch.tensor(theta_hat_list[j]) for j in range(index + 1, k)]
        theta_to_sigmom = (
            lambda x, index=index, solved_tail=solved_tail:
            Phi([x] + solved_tail, decay_list[index:], K_list[index], p)
        )
        theta_hat, _ = Fn_inverse(
            trace_moments, p, n, B_list[index], U_list[index:],
            theta_to_sigmom,
            thresh=thresh, clamp_max=clamp_max, clamp_min=clamp_min, seed=seed
        )
        if decay_list[index]=='step':
            taus = theta_hat[:int((K_list[index]+1)/2)]
            pis = np.append(theta_hat[int((K_list[index]+1)/2):],1-sum(theta_hat[int((K_list[index]+1)/2):]))
            idx_sorted = np.argsort(taus)[::-1]
            theta_hat = np.concatenate((taus[idx_sorted],pis[idx_sorted][:-1]))
        theta_hat_list[index] = theta_hat
    return theta_hat_list

def Fn_inverse_nested_deteqv(alpha_list,B_list,U_list,decay_list,p,thresh=0.005,clamp_max=20.0,clamp_min=-1.0):
    k=len(decay_list); n = B_list[0].shape[0]
    theta_hat_list = [None] * k
    K_list = [len(x) for x in alpha_list]

    for index in range(k - 1, -1, -1):
        trace_moments = np.array(alpha_list[index])
        solved_tail = [torch.tensor(theta_hat_list[j]) for j in range(index + 1, k)]
        theta_to_sigmom = (
            lambda x, index=index, solved_tail=solved_tail:
            Phi([x] + solved_tail, decay_list[index:], K_list[index], p)
        )
        theta_hat, _ = Fn_inverse(
            trace_moments, p, n, B_list[index], U_list[index:],
            theta_to_sigmom,
            thresh=thresh, clamp_max=clamp_max, clamp_min=clamp_min, seed = 0
        )
        if decay_list[index]=='step':
            taus = theta_hat[:int((K_list[index]+1)/2)]
            pis = np.append(theta_hat[int((K_list[index]+1)/2):],1-sum(theta_hat[int((K_list[index]+1)/2):]))
            idx_sorted = np.argsort(taus)[::-1]
            theta_hat = np.concatenate((taus[idx_sorted],pis[idx_sorted][:-1]))
        theta_hat_list[index] = theta_hat
    return theta_hat_list
    


def jacobian_Fn_nested(theta_list,B_list,U_list,decay_list,p,cache=None, verbose=True):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    
    theta_full = torch.cat(theta_list)
    n=B_list[0].shape[0]

    k = len(theta_list)
    K_list = []
    for r in range(k):
        K_list.append(len(theta_list[r]))
    
    if cache is not None:
        cache_folder_jac = 'cache/'+cache+'/'
        if not os.path.exists(cache_folder_jac):
            os.makedirs(cache_folder_jac)

    the_to_sigmom_functions = []
    for i in range(k):
        the_to_sigmom_functions.append(lambda x, i=i: Phi(x_to_list(x,K_list[i:]),decay_list[i:],K_list[i],p,jac=True))
    
    jacobian_F_list = []
    jacobian_F_full = torch.zeros((sum(K_list),sum(K_list)))
 
    true_moments = []

    for index in range(k-1,-1,-1):
        vprint("level ",index+1)
        if cache is not None:
            jacobian_Phi_path = os.path.join(cache_folder_jac, f"level_{index+1}_G_matrix_level_jacobian_Phi.pt")
            moments_path = os.path.join(cache_folder_jac, f"level_{index+1}_G_matrix_level_moments.pt")
            jacobian_Psi_path = os.path.join(cache_folder_jac, f"level_{index+1}_G_matrix_level_jacobian_Psi.pt")
            jacobian_F_path = os.path.join(cache_folder_jac, f"level_{index+1}_G_matrix_level_jacobian_F.pt")
            

        if cache is not None and all(os.path.exists(path) for path in [jacobian_Phi_path, moments_path, jacobian_Psi_path, jacobian_F_path]):
            # Load from cache
            vprint(f"Loading cached results for level {index+1}")
            jacobian_Phi = torch.load(jacobian_Phi_path)
            moments = torch.load(moments_path)
            true_moments.insert(0, moments.tolist())
            vprint("moments: ",moments.tolist())
            jacobian_Psi = torch.load(jacobian_Psi_path)
            jacobian_F = torch.load(jacobian_F_path)
            if index==k-1:
                jacobian_F_full[-sum(K_list[index:]):,-sum(K_list[index:]):] = jacobian_F
            else:
                jacobian_F_full[-sum(K_list[index:]):-sum(K_list[index+1:]),-sum(K_list[index:]):] = jacobian_F
            jacobian_F_list.insert(0,jacobian_F.tolist())
        else:
            vprint(the_to_sigmom_functions[index])
            key_to_pos_dict,Sig_mom_list,jacobian_Phi = the_to_sigmom_functions[index](theta_full[-sum(K_list[index:]):])
            vprint("jacobian_Phi: ",jacobian_Phi)

            Sig_mom_tensor = Sig_mom_list.clone().detach().requires_grad_(True)
            moments = Psi_n(Sig_mom_tensor, K_list[index], p, n, B_list[index], U_list[index:], key_to_pos_dict)
            vprint("moments: ",moments.tolist())

            jacobian_Psi = torch.autograd.functional.jacobian(lambda x: Psi_n(x,K_list[index], 
                                                                                p, n, B_list[index], 
                                                                                U_list[index:], 
                                                                                key_to_pos_dict),Sig_mom_tensor)
            vprint("jacobian_Psi: ",jacobian_Psi)
            jacobian_F = jacobian_Psi@jacobian_Phi
            if cache is not None:
                torch.save(jacobian_Psi, os.path.join(cache_folder_jac, f"level_{index+1}_G_matrix_level_jacobian_Psi.pt"))
                torch.save(jacobian_F, os.path.join(cache_folder_jac, f"level_{index+1}_G_matrix_level_jacobian_F.pt"))

            if index==k-1:
                jacobian_F_full[-sum(K_list[index:]):,-sum(K_list[index:]):] = jacobian_F
            else:
                jacobian_F_full[-sum(K_list[index:]):-sum(K_list[index+1:]),-sum(K_list[index:]):] = jacobian_F
            jacobian_F_list.insert(0,jacobian_F.tolist())
            
            vprint(f"jacobian_F: ",jacobian_F)
            
            
            vprint("#"*60)

    return jacobian_F_full
