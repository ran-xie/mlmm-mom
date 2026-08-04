import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt

def apply_vec(z,f,progress=False):
    '''
    Evaluate a function f over all z[i] for a vector z.

    Parameters
	----------
    z: torch.Tensor
    f: callable
    progress: bool
        If True, display a progress bar during evaluation.
    
    Returns 
    ----------
    res: torch.Tensor
        A vector of length len(z) containing the evaluated values f(z[i]).
    '''
    res=torch.complex(torch.zeros(z.shape[0]),torch.zeros(z.shape[0]))
    if progress:
        for i in tqdm(range(z.shape[0]), desc=progress): 
            res[i]=f(z[i])
    else:
        for i in range(z.shape[0]): 
            res[i]=f(z[i])
    return res 

def apply_grid(z1,z2,f,progress=False):
    '''
    Evaluate a bivariate function f over all pairs (z1[i], z2[j]) for vectors z1 and z2.

    Parameters
	----------
    z1: torch.Tensor
    z2: torch.Tensor
    f: callable
    progress: bool
        If True, display a progress bar during evaluation.
    
    Returns 
    ----------
    res: torch.Tensor
        A tensor of shape (len(z1), len(z2)) containing the evaluated values f(z1[i], z2[j]).
    '''
    res=torch.complex(torch.zeros((z1.shape[0],z2.shape[0])),torch.zeros((z1.shape[0],z2.shape[0])))
    if progress:
        for i in tqdm(range(z1.shape[0]), desc=progress):
            for j in range(z2.shape[0]):
                res[i,j]=f(z1[i],z2[j])
    else:
        for i in range(z1.shape[0]):
            for j in range(z2.shape[0]):
                res[i,j]=f(z1[i],z2[j])
    return res


def generate_owub_design(I, J=None, seed=1):
    '''
    One realization of a one way unbalanced full sibling design.

    Parameters
	----------
    I : int
        Number of families.
    J : list of int, optional
        Pre-specified number of siblings for each family, J = [J_1, ..., J_I].
        If None, the number of siblings is randomly generated as Unif{1,2}.
    seed : int, optional (default=1)
        Random seed used for generating S.

    Returns 
    ----------
    U_list : list of numpy.ndarray
        Membership matrices [U, I_n], where U assigns each individual to the corresponding family.
    B_list : list of numpy.ndarray
        Matrices [B_1, B_2] used to construct sum-of-squares matrices 
        S_nr = (1/p) Y^T B_r Y for each r.
    S : numpy.ndarray
        Number of siblings in each family.
    n : int
        Total number of individuals across all families.
    '''

    np.random.seed(seed)

    k = 2
    if J is None:
        J=np.random.binomial(size = I, n = 1, p = 0.5) + 1

    n=np.sum(J);
    
    U0 = np.ones((n, 1))
    U1=np.zeros([n,I])
    sum_=0
    for i in range(I):
        U1[sum_:(sum_+J[i]),i]=1
        sum_=sum_+J[i]    
    U2= np.eye(n)
    U_list = [U0,U1,U2]
    
    B_list = []
    for i in range(k):
        B_list.append(U_list[i+1]@np.linalg.inv(U_list[i+1].T@U_list[i+1])@U_list[i+1].T 
                      - U_list[i]@np.linalg.inv(U_list[i].T@U_list[i])@U_list[i].T)
    
    U_list = U_list[1:]
    return U_list, B_list, J, n

def plot_MOM_est(df_theta_hat, theta_true, figsize, x_ticks=False, xlabel=False, save = False):
    '''
    Plot histograms of method-of-moments estimators, with a vertical line 
    indicating the corresponding true population parameter.
    '''
    print(len(df_theta_hat)," replications")
    titles = df_theta_hat.columns
    fig, axes = plt.subplots(1, len(theta_true), figsize=figsize, sharey=True)
    axes[0].set_ylabel('Count', fontsize=12)
    for i,col in enumerate(df_theta_hat.columns):
        ax = axes[i]
        ax.hist(df_theta_hat[col], bins=10, alpha=0.5, color='grey')
        ax.axvline(theta_true[i], color='red', linestyle='--', linewidth=2.5) 
        ax.set_title(titles[i], fontsize=18)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        if x_ticks:
            ax.set_xlim(left = min(df_theta_hat[col].min()-0.5*df_theta_hat[col].std(),x_ticks[i][0]-0.25*(x_ticks[i][1]-x_ticks[i][0])),
                        right=max(df_theta_hat[col].max()+0.5*df_theta_hat[col].std(),x_ticks[i][-1]+0.25*(x_ticks[i][1]-x_ticks[i][0])))
            ax.set_xticks(x_ticks[i])
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()
    sim_bias = [round(x, 4) for x in (df_theta_hat.mean() 
                                      - np.array(theta_true)).tolist()]
    sim_sd = [round(x, 4) for x in df_theta_hat.std() ]
    return sim_bias,sim_sd


def plot_hist(df, mean, figsize, xlabel=False, save = False):
    '''
    Plot histograms of each column in a DataFrame with a vertical line at the specified mean.
    '''
    print(len(df)," replications")
    titles = df.columns
    fig, axes = plt.subplots(1, len(mean), figsize=figsize, sharey=True)
    axes[0].set_ylabel('count', fontsize=14)
    for i,col in enumerate(df.columns):
        ax = axes[i]
        ax.hist(df[col], bins=10, alpha=0.5, color='grey')
        ax.axvline(mean[i], color='red', linestyle='--', linewidth=2.5)  
        ax.set_title(titles[i], fontsize=20)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()
    return 

def qq_plots(df,figsize=False,save=False):
    '''
    Produce QQ plots for each column in a DataFrame.
    '''
    n_cols = df.shape[1]
    if not figsize:
        figsize=(4.5*n_cols,2.7)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize,sharex=True,sharey=True)

    for i, col in enumerate(df.columns):
        data = df[col].dropna().values
        sorted_data = np.sort(data)
        sorted_data = (sorted_data-np.mean(sorted_data))/np.std(sorted_data)
        n = len(sorted_data)
        probs = (np.arange(1, n + 1) - 0.5) / n
        theoretical_q = norm.ppf(probs)

        ax = axes[i]
        ax.plot(np.arange(-4,4,0.1), np.arange(-4,4,0.1), 'r--', linewidth=2)
        ax.plot(theoretical_q, sorted_data, 'o', color = "blue", markersize=2)
        ax.set_title(f'{col}', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlim(ax.get_ylim())
        ax.grid(True, linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Sample Quantiles', fontsize=15)
    fig.text(0.5, -0.04, 'Theoretical Quantiles', ha='center', fontsize=15)
    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches='tight')
    plt.show()
        
