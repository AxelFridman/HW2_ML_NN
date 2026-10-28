"""Part 4 and bonus: k augmented datasets (1,000 training points moved by Gaussian noise) scored by the best SVM."""
import math
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import logsumexp
from scipy.stats import rankdata
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.svm import SVC

from common import (AQUA, GENERATORS, INK, JOBS, M_AUG, MUTED, NAMES, OUT, SD, SPIRAL, THRESHOLD, R, arm, augment,
                    best_svm, crossing, decision, load, save, save_tex, scatter, stream, threshold)

K = 10_000                                         # augmented datasets per SD, k >> 1000
SWEEP, K_SWEEP = np.geomspace(0.01, 5, 30), 200    # from a tenth of the spiral's noise to past the clusters' gap
OVERFIT = dict(kernel='rbf', gamma=1000, C=1000)   # kernel 5x narrower than the noise: memorises the points
ARMS = [arm(np.linspace(0, SPIRAL['K'], 400), c) for c in (0, 1)]  # t-grid for the Bayes likelihood integral
METRICS = [('accuracy', 'acc', 100, '.1f'), ('F1', 'fOne', 1, '.3f'), ('AUC', 'auc', 1, '.3f')]


def chunked(f, X):
    """f on all points of the k datasets X (k x M x d), computed in parallel chunks."""
    chunks = np.array_split(X.reshape(-1, X.shape[-1]), math.ceil(X.shape[0] * X.shape[1] / 10_000))
    return np.concatenate(Parallel(n_jobs=JOBS)(delayed(f)(c) for c in chunks)).reshape(X.shape[:2])


def bayes(name, X, sd):
    """Log-likelihood ratio of class 1 to class 0 at X under part 1's generator with total noise SD sd."""
    if name == 'linear':
        return X[:, 0] + X[:, 1]        # proportional to it for two equal round Gaussians
    ll = [logsumexp(-euclidean_distances(X, a, squared=True) / (2 * sd ** 2), axis=1) for a in ARMS]
    return ll[1] - ll[0]


def predictors(models, name, sd):
    """Score function of each model and of the Bayes classifier at augmentation SD sd."""
    f = {key: partial(decision, m) for key, m in models.items()}
    f['bayes'] = partial(bayes, name, sd=np.hypot(GENERATORS[name][1]['noise'], sd))
    return f


def metrics(score, y):
    """Accuracy, F1 and AUC of each augmented dataset (one row each)."""
    pred = score > 0
    tp, fp, fn = (pred & (y == 1)).sum(1), (pred & (y == 0)).sum(1), (~pred & (y == 1)).sum(1)
    pos = y.sum(1)
    auc = ((rankdata(score, axis=1) * y).sum(1) - pos * (pos + 1) / 2) / (pos * (y.shape[1] - pos))  # Mann-Whitney
    return pd.DataFrame({'accuracy': (pred == y).mean(1), 'F1': 2 * tp / (2 * tp + fp + fn), 'AUC': auc})


tables, sweep, nums, shown = [], [], {}, {}
for d, name in enumerate(NAMES):
    Xtr, Xte, ytr, yte = load(name)
    models = {'best': best_svm(name).fit(Xtr, ytr)}
    if name == 'spiral':
        models['overfit'] = SVC(**OVERFIT).fit(Xtr, ytr)
        boundary = models['best']
    for key, m in models.items():
        nums[f'train{name}{key}'] = f'{100 * m.score(Xtr, ytr):.1f}'
        nums[f'test{name}{key}'] = f'{100 * m.score(Xte, yte):.1f}'
    for j, (level, sd) in enumerate(SD.items()):
        Xa, ya = augment(Xtr, ytr, sd, K, stream(4, d, j))
        for key, f in predictors({'best': models['best']}, name, sd).items():
            s = chunked(f, Xa)
            tables.append(metrics(s, ya).assign(dataset=name, level=level, model=key))
            if name == 'spiral' and key == 'best':
                shown[level] = Xa[0], ya[0], s[0] > 0
    for j, sd in enumerate(SWEEP):
        Xa, ya = augment(Xtr, ytr, sd, K_SWEEP, stream(4, d, 2, j))
        sweep.append({'dataset': name, 'sd': sd,
                      **{key: np.mean((chunked(f, Xa) > 0) == ya) for key, f in predictors(models, name, sd).items()}})
    print(name, 'done', flush=True)

res = pd.concat(tables)
summary = res.groupby(['dataset', 'model', 'level']).agg(['mean', 'std'])
summary.to_csv(f'{OUT}/augment.csv')
sweep = pd.DataFrame(sweep)
sweep.to_csv(f'{OUT}/augment_sweep.csv', index=False)
for (name, model, level), r in summary.iterrows():
    for metric, short, scale, fmt in METRICS:
        tag = f'{short}{name}{model}{level}'
        nums[tag] = f'{scale * r[(metric, "mean")]:{fmt}}'
        nums[tag + 'SD'] = f'{scale * r[(metric, "std")]:{fmt}}'
for name, g in sweep.groupby('dataset'):
    for key in ['best', 'bayes'] + (['overfit'] if name == 'spiral' else []):
        nums[f'star{name}{key}'] = f'{crossing(g.sd.values, g[key].values, log=True):#.2g}'
    nums[f'gapBayes{name}'] = f'{100 * (g.bayes - g.best).max():.1f}'

lim = (-R(SPIRAL['K']) - 0.6, R(SPIRAL['K']) + 0.6)
xx, yy = np.meshgrid(np.linspace(*lim, 300), np.linspace(*lim, 300))
f = decision(boundary, np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
f[np.hypot(xx, yy) > R(SPIRAL['K']) + 0.3] = np.nan     # beyond the data the boundary is arbitrary
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for i, (level, (X, y, pred)) in enumerate(shown.items()):
    ax, wrong = axes[i], pred != y
    scatter(ax, X, y, s=6, alpha=0.7)
    ax.scatter(*X[wrong].T, marker='x', s=14, color=INK, lw=0.8)
    ax.plot([], [], 'x', color=INK, label='misclassified')
    ax.contour(xx, yy, f, levels=[0], colors=INK, linewidths=0.8)
    ax.set(xlim=lim, ylim=lim,
           title=f'({"ab"[i]}) {level} SD, $\\sigma$ = {SD[level]}: accuracy {100 * (1 - wrong.mean()):.1f}%')
axes[0].legend(loc='lower left', fontsize=9)

ax = axes[2]
style = {('spiral', 'best'): dict(color=AQUA, label='spiral: best SVM'),
         ('spiral', 'overfit'): dict(color=AQUA, ls='--', lw=1.2, label='spiral: over-fitted'),
         ('linear', 'best'): dict(color=INK, label='linear: linear SVM')}
for (name, key), kw in style.items():
    g = sweep[sweep.dataset == name]
    ax.plot(g.sd, 100 * g[key], **kw)
for name, g in sweep.groupby('dataset'):
    ax.plot(g.sd, 100 * g.bayes, color=MUTED, ls=':', lw=1.5,
            label='Bayes (new points)' if name == 'linear' else None)
for j, sd in enumerate(SD.values()):
    ax.axvline(sd, color=MUTED, lw=0.8, alpha=0.5, label='low and high SD' if j == 0 else None)
threshold(ax, 0.011)
ax.set(xscale='log', ylim=(45, 101), xlabel='augmentation SD $\\sigma$',
       ylabel=f'mean accuracy over {K_SWEEP} datasets (%)', title='(c) accuracy against noise SD')
ax.legend(loc='lower left', fontsize=9)
save(fig, 'augment')
save_tex('augment', kAug=f'{K:,}', mAug=f'{M_AUG:,}', sdLow=SD['low'], sdHigh=SD['high'], kSweep=K_SWEEP,
         thr=f'{100 * THRESHOLD:.0f}', gammaOver=OVERFIT['gamma'], COver=OVERFIT['C'], **nums)
print(summary.round(4), sweep.round(3).to_string(), nums, sep='\n')
