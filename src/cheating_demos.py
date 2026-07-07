"""The two ways people fake a near-perfect football model — reproduced so you can
recognise them. Run: PYTHONPATH=src python src/cheating_demos.py"""
from __future__ import annotations

import numpy as np
import pandas as pd

import model as M

HOME, DRAW, AWAY = 0, 1, 2


def _softmax_fit(X, y, epochs=2000, lr=0.3, l2=0.0):
    mean, std = X.mean(0), X.std(0); std[std == 0] = 1
    Z = np.c_[np.ones(len(X)), (X - mean) / std]
    W = np.zeros((Z.shape[1], 3)); Y = np.eye(3)[y]
    for _ in range(epochs):
        p = np.exp(Z @ W - (Z @ W).max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        g = Z.T @ (p - Y) / len(Z); g[1:] += l2 * W[1:]; W -= lr * g
    return mean, std, W


def _predict(model, X):
    mean, std, W = model
    Z = np.c_[np.ones(len(X)), (X - mean) / std]
    p = np.exp(Z @ W - (Z @ W).max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    return p.argmax(1)


def _design():
    r = M.load_results()
    y = np.where(r["hs"] > r["as_"], HOME, np.where(r["hs"] < r["as_"], AWAY, DRAW))
    # a few honest features: importance, recency, neutral flag, goal totals baseline
    age = (M.WC_START - r["date"]).dt.days.to_numpy(float) / 365.0
    base = np.c_[r["imp"].to_numpy(float), age, (~r["neutral_flag"]).to_numpy(float)]
    teams = {t: i for i, t in enumerate(sorted(set(r["home"]) | set(r["away"])))}
    oh = np.zeros((len(r), len(teams)))
    for k, (h, a) in enumerate(zip(r["home"].map(teams), r["away"].map(teams))):
        oh[k, h] += 1; oh[k, a] -= 1  # team-identity: lots of free parameters
    gd = (r["hs"] - r["as_"]).to_numpy(float)
    return base, oh, gd, y


def overfitting_cliff():
    base, oh, gd, y = _design()
    rng = np.random.default_rng(7); perm = rng.permutation(len(y))
    tr, va = perm[:4000], perm[4000:6000]
    X = np.concatenate([base, oh], axis=1)
    print("  brakes (L2) | train acc | unseen acc")
    for l2 in [1.0, 0.1, 0.01, 0.0]:
        m = _softmax_fit(X[tr], y[tr], l2=l2)
        atr = (_predict(m, X[tr]) == y[tr]).mean()
        ava = (_predict(m, X[va]) == y[va]).mean()
        print(f"     {l2:<6} |  {atr:6.1%}  |  {ava:6.1%}")


def leakage():
    _, _, gd, y = _design()
    pred = np.where(gd > 0, HOME, np.where(gd < 0, AWAY, DRAW))  # gd IS the result
    print(f"  feeding the final goal difference -> accuracy {(pred == y).mean():.1%}")


if __name__ == "__main__":
    print("OVERFITTING CLIFF (remove brakes: train up, unseen down):")
    overfitting_cliff()
    print("\nLEAKAGE (a feature that is the answer):")
    leakage()
