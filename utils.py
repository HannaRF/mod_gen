import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split  # <-- Faltava esse!

# -----------------------------------------
# 3) Métricas
# -----------------------------------------
def compute_hsic(z):
    n,d = z.size()
    zc = z - z.mean(0, keepdim=True)
    D = torch.cdist(zc, zc)
    sigma = D.median()
    K = torch.exp(-D**2/(2*sigma**2))
    H = torch.eye(n, device=z.device) - 1/n
    hsic = (K @ H @ K @ H).trace()/((n-1)**2)
    return hsic.item()

def compute_quasi_ortho(z, eps=1e-3):
    Zf = F.normalize(z, p=2, dim=0)
    G = Zf.T @ Zf
    G.fill_diagonal_(0.)
    max_corr = G.abs().max().item()
    return (max_corr <= eps), max_corr

def compute_mig(z_np, factors_np):
    if len(np.unique(factors_np)) > 2:
        est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
        factors_np = est.fit_transform(factors_np)
    n, d = z_np.shape
    _, m = factors_np.shape
    mig_scores = []
    for j in range(m):
        mi = mutual_info_classif(z_np, factors_np[:, j])
        p, counts = np.unique(factors_np[:, j], return_counts=True)
        probs = counts / counts.sum()
        H = -np.sum(probs * np.log(probs + 1e-12))
        if H > 0:
            mi_sorted = np.sort(mi)[::-1]
            mig_scores.append((mi_sorted[0] - mi_sorted[1]) / H)
    return np.mean(mig_scores) if mig_scores else 0.0

def compute_sap(z_np, factors_np, test_size=0.3, random_state=0):
    if len(np.unique(factors_np)) > 2:
        est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
        factors_np = est.fit_transform(factors_np)

    sap_scores = []
    z_dim = z_np.shape[1]

    for j in range(factors_np.shape[1]):
        y = factors_np[:, j]
        if np.unique(y).size < 2:
            continue
        errs = []
        for i in range(z_dim):
            X = z_np[:, i].reshape(-1, 1)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            try:
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X_tr, y_tr)
                errs.append(1 - clf.score(X_te, y_te))
            except ValueError:
                continue
        if len(errs) < 2:
            continue
        errs = np.array(errs)
        idx = np.argsort(errs)
        sap_scores.append(errs[idx[1]] - errs[idx[0]])
    return np.mean(sap_scores) if sap_scores else 0.0