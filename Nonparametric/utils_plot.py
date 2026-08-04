import numpy as np
import matplotlib.pyplot as plt

def plot_compare(ppl_eigvals, d_oracle, d_hat, d_iso, save=None):
    p = len(ppl_eigvals)
    fs_label = 14
    fs_title = 15
    fs_tick = 12
    fs_legend = 12

    _, axes = plt.subplots(1, 3, figsize=(8, 3), sharey=True)

    d_list = [d_oracle, d_hat, d_iso]
    labels = ["Oracle", "Bona Fide", "Isotonic"]

    for i in range(3):
        ax = axes[i]
        ax.tick_params(axis="both", labelsize=fs_tick)
        ax.plot(np.arange(p), ppl_eigvals, ":", color="black", label="population")
        ax.plot(np.arange(p), d_list[i], color="b")
        ax.set_xlabel("Index", fontsize=fs_label)
        ax.set_title(labels[i], fontsize=fs_title, fontweight="medium")

    axes[0].legend(loc="upper right", fontsize=fs_legend)
    axes[0].set_ylabel("Eigenvalue", fontsize=fs_label)

    plt.tight_layout()
    if save is not None:
        plt.savefig(f"image/{save}.pdf", bbox_inches="tight")
    plt.show()

def plot_compare_bf_deteqv(d_hat, m_hat_Im, a1_hat_Im, d_check, m_check_Im, a1_check_Im, save=False):
    p=len(d_hat)
    fs_label = 14
    fs_title = 15
    fs_tick = 12
    fs_legend = 11

    labels = [r"$\hat{d}_n$", r"$\Im\hat{m}_n$", r"$\Im\hat{a}_n^{(1)}$"]
    bf_list = [d_hat, m_hat_Im, a1_hat_Im]
    deteqv_list = [d_check, m_check_Im, a1_check_Im]

    _, axes = plt.subplots(1, 3, figsize=(8, 3))
    for i in range(3):
        ax = axes[i]
        ax.plot(np.arange(p), bf_list[i], color="blue")
        ax.plot(np.arange(p), deteqv_list[i], ":", color="black", label="deterministic")
        ax.tick_params(axis="both", labelsize=fs_tick)
        ax.set_xlabel("Index", fontsize=fs_label)
        ax.set_title(labels[i], fontsize=fs_title)

    axes[0].legend(loc="upper right", fontsize=fs_legend)
    axes[0].set_ylabel("Eigenvalue", fontsize=fs_label)

    plt.tight_layout()
    if save:
        plt.savefig("image/" + save + ".pdf", bbox_inches="tight")
    plt.show()

def plot_compare_classical(ppl_eigvals, d_oracle, d_iso, Lambda_M, Lambda_R, save=None):
    p = len(ppl_eigvals)
    fs_label = 14
    fs_tick = 12
    fs_legend = 12

    _,axes = plt.subplots(1,2,figsize=(5.5,3),sharey=True)

    axes[0].plot(np.arange(p), ppl_eigvals, ':b', color='black', label='population')
    axes[0].scatter(np.arange(0,p,5),(Lambda_R)[np.arange(0,p,5)], label='REML',color='b',marker='+')
    axes[0].plot(np.arange(0,p,5),(Lambda_M)[np.arange(0,p,5)], label=r'$\text{MANOVA}$',color='r')
    axes[0].set_ylabel('Eigenvalue', fontsize=fs_label)

    axes[1].plot(np.arange(p),ppl_eigvals, ':b', color='black')
    axes[1].scatter(np.arange(0,p,5),(d_oracle)[np.arange(0,p,5)], label=r'$\text{Oracle NP}$',color='b',marker='+')
    axes[1].plot(np.arange(p),d_iso, label=r'$\text{Bona fide NP}$',color='r')

    for ax in axes:
        ax.legend(loc='upper right',fontsize=fs_legend)
        ax.set_xlabel('Index', fontsize=fs_label)
        ax.tick_params(axis="both", labelsize=fs_tick)

    plt.tight_layout()
    if save is not None:
        plt.savefig(f"image/{save}.pdf", bbox_inches="tight")
    plt.show()