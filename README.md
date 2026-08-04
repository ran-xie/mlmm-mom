# Method of Moments Estimation of High-Dimensional Covariance Matrices within Random Effects Models 

## Installation
`mlmm-mom` depends on our companion package [`spectral-clt`](https://github.com/ran-xie/spectral-clt). Install both as follows:
```bash
git clone https://github.com/ran-xie/spectral-clt.git
git clone https://github.com/ran-xie/mlmm-mom.git
```

## Model Setup
We study multi-trait linear mixed effects models in a high-dimensional regime, where the number of traits $p$ may be comparable to or exceed the number of individuals $n$.

### Model MLMM
(i) Consider a $k$-level multi-trait linear mixed model (MLMM) of the form

$$ Y = X\beta + \sum_{r=1}^k U_r \alpha_r, \quad \alpha_r \sim N(0, \text{Id}_{I_r} \otimes \Sigma_r),$$

where $Y \in \mathbb{R}^{n \times p}$ is the observed trait matrix, $X \in \mathbb{R}^{n \times I_0}$ is the fixed effects design matrix, $\beta \in \mathbb{R}^{I_0 \times p}$ is the fixed effect coefficient matrix, $U_r \in \mathbb{R}^{n \times I_r}$ are known design matrices, and $\alpha_r \in \mathbb{R}^{I_r \times p}$ are independent random effect matrices.

(ii) Let  $B \in \mathbb{R}^{n \times n}$ be a non-zero positive semi-definite matrix that satisfies $BX = 0$. We study quadratic
  forms and empirical moments

$$ S_n = p^{-1} Y^T B Y, \qquad\hat{\mu}_{nl} = p^{-1} \text{Tr}(S_n^l) $$

Let $\hat\mu_n = (\hat\mu_{n1}, \ldots, \hat{\mu}_{nK})$ be the first $K$ empirical moments of $S_n$.

### Estimating $\Sigma_1,\ldots,\Sigma_k$
We assume that the eigenvalue distribution of each covariance matrix $\Sigma_r$ is parameterized by a finite-dimensional parameter vector $\theta_r \in \mathbb{R}^{k_r}$. We show that in a high-dimensional regime where $n, p, I_1, \cdots, I_k \to \infty$, there exists an explicit function $F_n$ such that $\hat\mu_n=F_n(\theta)+o_{a.s.}(1)$,
which naturally leads to the following method of moments estimator

$$ \hat{\theta}_n={F}_n^{-1}(\hat{\mu}_n).$$

Assuming that $\Sigma_1,\ldots,\Sigma_k$ are simultaneously diagonalizable, along with some mild regularity conditions, we show that the proposed MoM estimator is consistent and asymptotically normal: there exist  $P_n\in\mathbb{R}^{K\times K}$ and $Q_n\in\mathbb{R}^K$, such that

$$ P_n^{-1/2}\left[p\left(\hat{\theta}_n-\theta\right)-Q_n\right]\rightarrow_d N(0,\text{Id}),$$

providing a principled approach to statistical inference for the spectral parameters of high-dimensional covariance components in multi-trait mixed linear models.

## Usage
### General Designs
Given an $n$ by $p$ obeservation matrix `Y`, a design matrix `X`, a list of $k$ incidence matrices `U_list=[U_1,...,U_k]`, a list of parametric decay models `decay_list = [decay_1,...,decay_k]`, a list of number of parameters in each model `K_list = [K_1,...,K_k]` and a determinsitic matrix `B` fixed a priori, we can solve for the MoM estimators `theta_hat` containing the parameters for each decay model, and numerically evaluate its asymptotic bias ($Q_n$) and covariance ($P_n$).
```python
from Parametric.Fn import Fn_inverse_joint
from Parametric.clt_MoM import clt_joint
theta_hat_list = Fn_inverse_joint(Y,B,U_list,decay_list,K_list)
bias,cov = clt_joint(theta_list,B,U_list,decay_list,p)
```

### Nested Designs
For common nested designs (with definitions in Section 2.3), we can solve for each $\theta_r$ parameterizing each $\Sigma_r$ sequentially, improving computational efficiency and accuracy.
```python
from Parametric.Fn_nested import Fn_inverse_nested
from Parametric.clt_MoM import clt_nested
theta_hat_list = Fn_inverse_nested(Y,B_list,U_list,decay_list,K_list)
bias,cov = clt_nested(theta_list,B_list,U_list,decay_list,p)
```

For a simplified version of the algorithm specialized to full sibling models, see 'experiments/fullsib.ipynb'.
In this notebook, we also demonstrate the empirical fix and oracle fix procedures.

## Reproducible Research
The code available under 'experiments/' in the repository replicates the experimental results in our paper.  

In 'experiments/compare_estimators.ipynb', we comare the classical MANOVA and REML estimators with the proposed parametric method-of-moments estimators and the nonparametric shrinkage estimator from our companion work [2], reproducing Figure 1.1 in [1].

## References
```bibtex
@article{MoM2026, 
  title={Method of Moments Estimation of High-Dimensional Covariance Using a Parametric Model}, 
  author={Iain M. Johnstone and Yuchen Wu and Ran Xie},
  year={2026},
  eprint={2608.01590},
  archivePrefix={arXiv},
  primaryClass={stat.ME},
  url={https://arxiv.org/abs/2608.01590}, 
}
```
[1] Xie R. (2026) Estimating covariance matrices within high-dimensional random effects models (Doctoral dissertation, Stanford University). Available at: https://doi.org/10.25740/nx184bp5557.

[2] Xie, R., & Johnstone, I. Nonparametric shrinkage estimation of high-dimensional variance component matrices. Manuscript in preparation.

[3] Xie, R., & Johnstone, I. (2024). CLT for Linear Spectral Statistics in High-Dimensional Random Effects Models. arXiv preprint arXiv:2406.03719.