import skccm as ccm
from skccm.utilities import train_test_split
import warnings
import pandas as pd
import numpy as np


def CrossCorr(datax, datay, maxlag=52):
    """
    Find the lag (1…maxlag−1) that maximizes the absolute Spearman correlation
    between datax and datay shifted by that lag.
    Returns: (best_lag, best_correlation)
    """
    dx = pd.Series(datax)
    dy = pd.Series(datay)
    best_corr = 0.0
    best_lag = 1
    for lag in range(1, maxlag):
        c = abs(dx.corr(dy.shift(lag), method="spearman"))
        if c > best_corr:
            best_corr = c
            best_lag = lag
    return best_lag, best_corr


def ccm_test(x1, x2, maxlag=52):
    """
    Identify the optimal lag and embedding dimension for CCM between x1 (target)
    and x2 (predictor). Returns (lag, embed, max_corr).
    """
    warnings.filterwarnings("ignore")

    # 1) pick lag via Spearman Cross‐correlation
    lag, _ = CrossCorr(x1, x2, maxlag=maxlag)

    best_score = 0

    # 2) sweep embedding dimension
    for E in range(2, 6):
        e1 = ccm.Embed(x1)
        e2 = ccm.Embed(x2)
        X1 = e1.embed_vectors_1d(lag, E)
        X2 = e2.embed_vectors_1d(lag, E)

        x1_tr, x1_te, x2_tr, x2_te = train_test_split(X1, X2, percent=0.9)
        lib_lens = np.unique(np.linspace(10, len(x1_tr), num=20, dtype=int))

        model = ccm.CCM(score_metric="corrcoef")
        model.fit(x1_tr, x2_tr)
        model.predict(x1_te, x2_te, lib_lengths=lib_lens)
        sc1, _ = model.score()  # sc1 is the array of corrcoefs for X1→X2

        if max(sc1) > best_score:
            best_score = max(sc1)
            best_embed = E

    return lag, best_embed, best_score


def choose_preds_ccm(df, target_col, top_n=5, maxlag=52):
    """
    Rank all columns in df (except target_col) by their CCM causality score
    with target_col, and return the top_n predictors.
    """
    # 1) drop any rows with NaNs in target or predictors
    preds = [c for c in df.columns if c != target_col]
    df_clean = df.dropna(subset=[target_col] + preds).reset_index(drop=True)

    # 2) scale target to [-1,1]
    y = df_clean[target_col].values
    y_scaled = y / np.max(np.abs(y))

    results = []
    for pred in preds:
        x = df_clean[pred].values
        # normalize predictor to [0,1]
        x_norm = (x - x.min()) / (x.max() - x.min())

        lag, embed, score = ccm_test(y_scaled, x_norm, maxlag=maxlag)
        results.append({"predictor": pred, "lag": lag, "embed": embed, "score": score})

    # 3) sort by descending CCM score and return top_n
    return sorted(results, key=lambda d: d["score"], reverse=True)[:top_n]
