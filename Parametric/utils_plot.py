import os,re,ast
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import MaxNLocator



def read_MoM_cache(cache_dir):
    with open(cache_dir, 'r') as f:
        lines = f.readlines()
        pattern = re.compile(r"(\d+),\[(.*?)\]")
        data = []
        for line in lines:
            match = pattern.match(line.strip())
            if match:
                seed = int(match.group(1))  # Extract first number
                param_list = ast.literal_eval(f"[{match.group(2)}]")  # Convert the list string to a Python list
                data.append([seed, param_list])
    df_theta_hat = pd.DataFrame(data, columns=['seed', 'parameters'])
    params_df = pd.DataFrame(df_theta_hat['parameters'].tolist(), index=df_theta_hat.index)
    df_theta_hat = pd.concat([df_theta_hat.drop(columns=['parameters']), params_df], axis=1)
    df_theta_hat = df_theta_hat.sort_values(by='seed')
    df_theta_hat = df_theta_hat.reset_index(drop=True)
    df_theta_hat = df_theta_hat.drop('seed',axis=1)
    return df_theta_hat

def plot_MOM_est(cache, titles, theta_true, figsize, save = False):
    df_theta_hat = read_MoM_cache(cache)
    df_theta_hat.columns = titles
    print(len(df_theta_hat)," replications")
    _, axes = plt.subplots(1, len(theta_true), figsize=figsize, sharey=True)
    for i,col in enumerate(df_theta_hat.columns):
        ax = axes[i]
        ax.hist(df_theta_hat[col], bins=10, alpha=0.5, color='grey')
        ax.axvline(theta_true[i], color='red', linestyle='--', linewidth=2.5)  # Add vertical red line at mean
        ax.set_title(titles[i], fontsize=20)
        #ax.xaxis.set_major_locator(LinearLocator(3))
        xmin, xmax = ax.get_xlim()
        xlow = xmin+(xmax-xmin)/10
        xhigh = xmax-(xmax-xmin)/10
        xticks = np.linspace(xlow, xhigh, 3)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.2f}" for x in xticks])
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()
    sim_bias = [round(x, 4) for x in (df_theta_hat.mean() 
                                      - np.array(theta_true)).tolist()]
    print("sim bias: ", sim_bias)
    sim_sd = [round(x, 4) for x in df_theta_hat.std() ]
    print("sim sd: ", sim_sd)
    sim_corr =  df_theta_hat.corr()
    return sim_bias,sim_sd, sim_corr


def qq_plots(cache, titles,figsize=False,save=False):
    df_theta_hat = read_MoM_cache(cache)
    print(len(df_theta_hat)," replications")
    n_cols = df_theta_hat.shape[1]
    if not figsize:
        figsize=(4.5*n_cols,2.7)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize,sharex=True,sharey=True)

    for i, col in enumerate(df_theta_hat.columns):
        data = df_theta_hat[col].dropna().values
        sorted_data = np.sort(data)
        sorted_data = (sorted_data-np.mean(sorted_data))/np.std(sorted_data)
        n = len(sorted_data)
        probs = (np.arange(1, n + 1) - 0.5) / n
        theoretical_q = norm.ppf(probs)

        ax = axes[i]
        ax.plot(np.arange(-4,4,0.1), np.arange(-4,4,0.1), 'r--', linewidth=2)
        ax.plot(theoretical_q, sorted_data, 'o', color = "blue", markersize=2)
        ax.set_title(titles[i], fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlim(ax.get_ylim())
        ax.grid(True, linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Sample Quantiles', fontsize=15)
    fig.text(0.5, -0.04, 'Theoretical Quantiles', ha='center', fontsize=15)
    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches='tight')
    plt.show()
        

def calc_intervals(df, typeconf="normal"):
    '''
    Calculates 99% normal approximation confidence interval.
    '''
    intervals = {}
    if typeconf=="normal":
        for col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - 2.575 * std #1.645
            upper_bound = mean + 2.575 * std
            intervals[col] = (lower_bound, upper_bound)
    elif typeconf=="emp_quantile":
        for col in df.columns:
            lower_bound = df[col].quantile(0.005)
            upper_bound = df[col].quantile(0.995)
            intervals[col] = (lower_bound, upper_bound)
    else:
        print("Unknown confidence interval type")
    return intervals


def plot_conf_bias_sd(cache,titles,theta_full,bias_theory, sd_theory,save=False,bias_xlim=False,sd_xlim=False):
    '''
    Plots 99% normal approximation confidence interval.
    '''
    df_theta_hat = read_MoM_cache(cache)
    df_theta_hat.columns = titles

    bias_list_full = []
    sd_list_full = []

    group_size=50
    sol_list = df_theta_hat.to_numpy()
    for j in range(0, len(sol_list), group_size):
        group = sol_list[j:j + group_size,:]
        bias_list = []
        sd_list = []
        for i in range(len(theta_full)):
            param_hat=group[:,i]
            param_bias=np.mean(param_hat)-np.array(theta_full)[i]
            param_sd = np.std(param_hat)
            bias_list.append(param_bias)
            sd_list.append(param_sd)
        bias_list_full.append(bias_list)
        sd_list_full.append(sd_list)
    bias_list_full = pd.DataFrame(bias_list_full,columns=df_theta_hat.columns)

    intervals_bias_normal = calc_intervals(bias_list_full,typeconf="normal")

    sd_list_full = pd.DataFrame(sd_list_full,columns=df_theta_hat.columns)
    intervals_sd_normal = calc_intervals(sd_list_full,typeconf="normal")

    fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=False)
    # === First plot: Bias ===
    ax_bias = axs[0]
    y_positions = np.arange(len(theta_full))

    for i, label in enumerate(df_theta_hat.columns):
        interval = intervals_bias_normal[label]
        ax_bias.fill_betweenx([i - 0.2, i + 0.2], interval[0], interval[1], color='grey', alpha=0.5)
        if len(bias_theory)>0:
            ax_bias.scatter(bias_theory[i], [i], color='red', s=100, zorder=5, marker='x')

    ax_bias.set_yticks(y_positions)
    ax_bias.set_yticklabels(df_theta_hat.columns, fontsize=12)
    ax_bias.set_xlabel("Bias", fontsize=14)
    ax_bias.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax_bias.invert_yaxis()
    if bias_xlim:
        ax_bias.set_xlim(bias_xlim)

    # === Second plot: SD ===
    ax_sd = axs[1]

    for i, label in enumerate(df_theta_hat.columns):
        interval = intervals_sd_normal[label]
        ax_sd.fill_betweenx([i - 0.2, i + 0.2], interval[0], interval[1], color='grey', alpha=0.5)
        if len(sd_theory)>0:
            ax_sd.scatter([sd_theory[i]], [i], color='red', s=100, zorder=5, marker='x')

    ax_sd.set_yticks(y_positions)
    ax_sd.set_yticklabels(df_theta_hat.columns, fontsize=12)  # Hide y-tick labels on right plot to avoid repetition
    ax_sd.set_xlabel("SD", fontsize=14)
    ax_sd.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax_sd.invert_yaxis()
    if sd_xlim:
        ax_sd.set_xlim(sd_xlim)
    if save:
        plt.savefig(save, bbox_inches='tight')
    #plt.tight_layout()
    plt.show()

def plot_compare(Lambda1,Lambda1_est,labels,save = None,label='Population',exp_conf=None,quad_conf=None,ylim=None):
    color_main = 'blue'       
    color_true = '#b2182b'       
    lw_main = 1.5
    lw_reml = 1.5
    fontsize_title = 17
    fontsize_ylabel = 15
    band_alpha = 0.28
    color_band = '#2166ac'

    p=len(Lambda1)
    fig, axes = plt.subplots(1, len(Lambda1_est), figsize=(2.4*len(Lambda1_est), 3), sharey=True)
    x = np.arange(1, p + 1)

    for i in range(len(Lambda1_est)):
        axes[i].plot(x, Lambda1, color=color_true, linestyle='--', linewidth=lw_reml, alpha=0.85, label=label)
        axes[i].plot(x, Lambda1_est[i], color=color_main, linewidth=lw_main)
        axes[i].set_title(labels[i], fontsize=fontsize_title, fontweight='medium')
        axes[i].set_xlabel('Index', fontsize=fontsize_ylabel)

    axes[0].set_ylabel('Eigenvalue', fontsize=fontsize_ylabel)
    if ylim is not None:
        axes[0].set_ylim(ylim)
    if exp_conf is not None:
        lower_exp,upper_exp =exp_conf
        lower_quad,upper_quad =quad_conf
        axes[-2].fill_between(x, lower_exp, upper_exp, alpha=band_alpha, color=color_band)
        axes[-1].fill_between(x, lower_quad, upper_quad, alpha=band_alpha, color=color_band)

    axes[0].legend(loc='upper right', fontsize=12, framealpha=0.95)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.5)
        ax.grid(True, color='lightgray', linestyle='-', alpha=0.6)
        ax.tick_params(axis='both', labelsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))

        fig.tight_layout(rect=[0, 0, 1, 0.92])
    else:
        fig.tight_layout()
    if save is not None:
        plt.savefig(save+'.pdf', bbox_inches='tight')
    plt.show()
