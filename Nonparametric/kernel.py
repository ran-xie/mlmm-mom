import numpy as np

def kernel_density(Lambda, n, T=None, eps=0):
    """
    Given empirical eigenvalues Lambda and weights T, 
    compute kernel smoothed density f_hat and its Hilbert transform Hf_hat.
    """
    L = np.tile(Lambda, (Lambda.shape[0], 1)).T
    h = n ** (-1/3)  
    H = h * np.maximum(np.abs(L.T), eps) 
    if T is not None:
        Lt = np.tile(T, (Lambda.shape[0], 1)).T
        Ht = H/Lt.T
    x = (L - L.T) / H

    if T is not None:
        f_hat = (3 / (4 * np.sqrt(5))) * np.mean(np.maximum(1 - x**2 / 5, 0) / Ht, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_term = np.log(np.abs((np.sqrt(5) - x) / (np.sqrt(5) + x)))
            Hftemp = (-3 / (10 * np.pi)) * x + (3 / (4 * np.sqrt(5) * np.pi)) * (1 - x**2 / 5) * log_term

        mask = np.isclose(np.abs(x), np.sqrt(5))
        Hftemp[mask] = (-3 / (10 * np.pi)) * x[mask]

        Hf_hat = np.mean(Hftemp / Ht, axis=1)
        return f_hat,Hf_hat
    f_hat = (3 / (4 * np.sqrt(5))) * np.mean(np.maximum(1 - x**2 / 5, 0) / H, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_term = np.log(np.abs((np.sqrt(5) - x) / (np.sqrt(5) + x)))
        Hftemp = (-3 / (10 * np.pi)) * x + (3 / (4 * np.sqrt(5) * np.pi)) * (1 - x**2 / 5) * log_term

    mask = np.isclose(np.abs(x), np.sqrt(5))
    Hftemp[mask] = (-3 / (10 * np.pi)) * x[mask]

    Hf_hat = np.mean(Hftemp / H, axis=1)
    return f_hat,Hf_hat


def select_eps_cv(Lambda, n, eps_grid=None, jitter=1e-12):
    """
    Choose eps by leave-one-out log-likelihood Cross Validate.
    """
    m = Lambda.shape[0]
    h = n ** (-1/3)

    L = np.tile(Lambda, (m, 1)).T  

    best_eps = None
    best_obj = np.inf

    ax = np.abs(Lambda)
    if eps_grid is None:
        q = np.linspace(0.01, 0.30, 25)
        eps_grid = np.quantile(ax[ax > 0], q) if np.any(ax > 0) else np.array([1e-6])


    for eps in eps_grid:
        H = h * np.maximum(np.abs(L.T), eps)   # H[i,j] = h*max(|Lambda[j]|,eps)

        x = (L - L.T) / H

        K = (3 / (4 * np.sqrt(5))) * np.maximum(1 - x**2 / 5, 0) / H

        np.fill_diagonal(K, 0.0)

        ftilde_loo = np.sum(K, axis=1) / (m - 1)

        # CV objective: minimize negative LOO log-likelihood
        obj = -np.mean(np.log(ftilde_loo + jitter))

        if obj < best_obj:
            best_obj = obj
            best_eps = eps

    return best_eps

