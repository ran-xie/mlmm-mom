import torch

def enumerate_solutions(k:int, l:int, current=None, solutions=None):
    """
    Enumerate all nonnegative integer solutions to:
        a_1 + ... + a_k = l
    The number of solutions is C(l + k - 1, k - 1).
    """
    if current is None:
        current = []
    if solutions is None:
        solutions = []  

    if len(current) == k - 1:
        current.append(l - sum(current))
        if current[-1] >= 0:
            solutions.append(current.copy())  
        current.pop()  # Backtrack
        return solutions

    for i in range(l + 1):
        current.append(i)
        enumerate_solutions(k, l, current, solutions)
        current.pop()  # Backtrack

    return solutions 


def theta_to_Sig_diag(theta, p: int, decay: str = "linear"):
    """
    Convert decay parameters into a length-p diagonal spectrum.
    Supported decays:
    - linear: theta = [tau, pi]
    - quadratic: theta = [tau, pi]
    - exp: theta = [tau1, tau2]
    - step: theta = [tau_1, ..., tau_m, pi_1, ..., pi_{m-1}]
    """
    theta = torch.as_tensor(theta).clone()

    if decay == "linear":
        eps=torch.tensor(1e-6)
        Sigma_diag = []
        for i in range(p):
            Sigma_diag.append(torch.maximum(torch.tensor(0.),theta[0]*(1-i/p*torch.maximum(theta[1],eps))))
    elif decay == "quadratic":
        eps=torch.tensor(1e-6)
        Sigma_diag = []
        for i in range(p):
            Sigma_diag.append(torch.maximum(torch.tensor(0.),theta[0]*(1-i/p*torch.maximum(theta[1],eps)))**2)
    elif decay == 'exp':
        Sigma_diag = []
        for i in range(p):
            Sigma_diag.append(theta[0]*torch.exp(-i/p*theta[1]))                                                    
    elif decay == "step":
        k = int((len(theta)+1)/2)
        tau = theta[0:k]
        pi_ori = theta[k:]
        pi_m = 1-torch.sum(pi_ori)
        pi = torch.minimum(torch.tensor(1.),torch.cat((pi_ori, torch.tensor([pi_m]))))
        if torch.isnan(pi).any():
            print("NaN values found in pi:", pi)
            print("theta: ",theta)
        Sigma_diag = []
        for i in range(len(tau)):
            Sigma_diag.extend([tau[i]]*int(p*pi[i]))
        if len(Sigma_diag)<p:
            Sigma_diag.extend([torch.tensor(tau[-1],dtype=torch.float32)]*(p-len(Sigma_diag)))
        elif len(Sigma_diag)>p:
            Sigma_diag = Sigma_diag[:p]
        Sigma_diag = sorted(Sigma_diag,reverse=True)
    else:
        print("ERROR decay function")
        return
    return torch.stack(Sigma_diag)

def Phi(theta_list_original,decay_list,L,p,jac=False): 
    """
    Compute the Phi function that maps the parameters theta to mixed population moments.

    Parameters
    ----------
    theta_list_original:
        List of theta parameter tensors, one for each Sigma.
    decay_list:
        List of decay types. Must have the same length as theta_list_original.
    L:
        Maximum total moment order.
    p:
        Dimension of each Sigma.
    jac:
        If True, also return the Jacobian of Phi.
    """
    k=len(theta_list_original)
    K = sum([len(theta) for theta in theta_list_original])
    x = torch.cat(theta_list_original).detach().clone().requires_grad_(True)
    step_binary = [1 if decay == "step" else 0 for decay in decay_list]
    ifstep = True if sum(step_binary)>0 else False
    step_indices = [i for i, val in enumerate(step_binary) if val == 1]
    expanded_binary = []
    for idx, theta in enumerate(theta_list_original):
        expanded_binary.extend([step_binary[idx]] * len(theta))
    expanded_binary_tensor = torch.tensor(expanded_binary, dtype=torch.bool)

    theta_list =[torch.zeros(len(theta_r)) for theta_r in theta_list_original]
    start_idx = 0
    for r in range(k):
        end_idx = start_idx + len(theta_list_original[r])
        theta_list[r]=x[start_idx:end_idx]
        start_idx = end_idx
    
    key_to_pos_dict = {}
    Sig_mom = []
    i=0
    Sigma_diag_list = []
    J_Phi_step = []

    for r in range(k):
        Sigma_diag_list.append(theta_to_Sig_diag(theta_list[r],p,decay_list[r]))
            
    for l in range(1, L + 1):
        all_solutions = enumerate_solutions(k, l)
        for solution in all_solutions:
            solution_tuple = tuple(solution)
            key_to_pos_dict[solution_tuple] = i
            mat = torch.ones(p) #=1
            for r in range(k):
                mat = mat*torch.pow(Sigma_diag_list[r],solution[r])
            Sig_mom.append(torch.sum(mat)/p)
            i+=1
            if ifstep and jac:
                jac_step = []
                for r_step in step_indices:
                    theta_r = theta_list[r_step]
                    k_r = int((len(theta_r)+1)/2)
                    tau_r = theta_r[0:k_r]
                    pi_r = torch.minimum(torch.tensor(1.),torch.tensor(theta_r[k_r:]))
                    
                    p_pi_r = torch.tensor([int(pi*p) for pi in pi_r])
                    cumsum_begin = torch.cat([torch.tensor([0]), torch.cumsum(p_pi_r, dim=0)]).to(torch.int) # the first index of each step
                    cumsum_end = torch.cat([torch.cumsum(p_pi_r, dim=0)-1,torch.tensor([p-1])]).to(torch.int) # the last index of each step
                    
                    if solution[r_step]==0:
                        jac_step.extend([0]*len(theta_r))
                    else:
                        for idx_tau_r_i,tau_r_i in enumerate(tau_r):
                            if tau_r_i==0:
                                if solution[r_step]>1:
                                    jac_step.extend([0])
                                else:
                                    mat0 = torch.ones(p) 
                                    for r0 in range(k):
                                        if r0!= r_step:
                                            mat0 = mat0*torch.pow(Sigma_diag_list[r0],solution[r0])
                                    jac_step.extend([1/p*torch.sum(mat0[cumsum_begin[idx_tau_r_i]:(cumsum_end[idx_tau_r_i]+1)])])
                            else:
                                jac_step.extend([1/p*solution[r_step]/tau_r_i*torch.sum(mat[cumsum_begin[idx_tau_r_i]:(cumsum_end[idx_tau_r_i]+1)])])
                        for idx_pi_r_i,pi_r_i in enumerate(pi_r):
                            positives = torch.sum(mat[cumsum_end[idx_pi_r_i:-1]])
                            negatives = torch.sum(mat[cumsum_begin[idx_pi_r_i+1:]])
                            jac_step.extend([positives-negatives])
                
                
                J_Phi_step.append(jac_step)


    res = torch.stack(Sig_mom)
    if not jac:
        return key_to_pos_dict,res

    if ifstep:
        if sum(step_binary)==len(step_binary):
            J_Phi = torch.tensor(J_Phi_step)
        else: 
            J_Phi = torch.zeros((len(J_Phi_step),K))
            J_Phi[:,expanded_binary_tensor] = torch.tensor(J_Phi_step)
            gradients = []
            for scalar in res:
                grad = torch.autograd.grad(scalar, x, retain_graph=True, create_graph=True)[0]
                gradients.append(grad)
            jacobian = torch.stack(gradients)
            J_Phi[:,~expanded_binary_tensor] = jacobian[:,~expanded_binary_tensor]
    else:
        gradients = []
        for scalar in res:
            grad = torch.autograd.grad(scalar, x, retain_graph=True, create_graph=True)[0]
            gradients.append(grad)
        J_Phi = torch.stack(gradients)
    
    return key_to_pos_dict,res,J_Phi




