"""Part 4 and bonus: k augmented datasets (1,000 training points moved by Gaussian noise) scored by the best SVM."""
import math
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.svm import SVC

from common import (AQUA, INK, JOBS, M_AUG, MUTED, NAMES, OUT, SD, SPIRAL, THRESHOLD, R, augment, best_svm, crossing,
                    decision, load, save, save_tex, scatter, stream, threshold)

K = 10_000                                         # augmented datasets per SD, k >> 1000
SWEEP, K_SWEEP = np.geomspace(0.01, 5, 30), 200    # from a tenth of the spiral's noise to past the clusters' gap
OVERFIT = dict(kernel='rbf', gamma=1000, C=1000)   # kernel 5x narrower than the noise: memorises the points
METRICS = [('accuracy', 'acc', 100, '.1f'), ('F1', 'fOne', 1, '.3f'), ('AUC', 'auc', 1, '.3f')]


def chunked(f, X):
    """f on all points of the k datasets X (k x M x d), computed in parallel chunks."""
    chunks = np.array_split(X.reshape(-1, X.shape[-1]), math.ceil(X.shape[0] * X.shape[1] / 10_000))
    return np.concatenate(Parallel(n_jobs=JOBS)(delayed(f)(c) for c in chunks)).reshape(X.shape[:2])


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
        s = chunked(partial(decision, models['best']), Xa)
        tables.append(metrics(s, ya).assign(dataset=name, level=level))
        if name == 'spiral':
            shown[level] = Xa[0], ya[0], s[0] > 0
    for j, sd in enumerate(SWEEP):
        Xa, ya = augment(Xtr, ytr, sd, K_SWEEP, stream(4, d, 2, j))
        sweep.append({'dataset': name, 'sd': sd,
                      **{key: np.mean((chunked(partial(decision, m), Xa) > 0) == ya) for key, m in models.items()}})
    print(name, 'done', flush=True)

res = pd.concat(tables)
summary = res.groupby(['dataset', 'level']).agg(['mean', 'std'])
summary.to_csv(f'{OUT}/augment.csv')
sweep = pd.DataFrame(sweep)
sweep.to_csv(f'{OUT}/augment_sweep.csv', index=False)
for (name, level), r in summary.iterrows():
    for metric, short, scale, fmt in METRICS:
        nums[f'{short}{name}{level}'] = f'{scale * r[(metric, "mean")]:{fmt}}'
        nums[f'{short}{name}{level}SD'] = f'{scale * r[(metric, "std")]:{fmt}}'
for name, g in sweep.groupby('dataset'):
    for key in ['best'] + (['overfit'] if name == 'spiral' else []):
        nums[f'star{name}{key}'] = f'{crossing(g.sd.values, g[key].values, log=True):#.2g}'

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

ax = axes[2]
style = {('spiral', 'best'): dict(color=AQUA, label='spiral: best SVM'),
         ('spiral', 'overfit'): dict(color=AQUA, ls='--', lw=1.2, label='spiral: over-fitted'),
         ('linear', 'best'): dict(color=INK, label='linear: linear SVM')}
for (name, key), kw in style.items():
    g = sweep[sweep.dataset == name]
    ax.plot(g.sd, 100 * g[key], **kw)
for j, sd in enumerate(SD.values()):
    ax.axvline(sd, color=MUTED, lw=0.8, alpha=0.5, label='low and high SD' if j == 0 else None)
threshold(ax, 0.011)
ax.set(xscale='log', ylim=(45, 101), xlabel='augmentation SD $\\sigma$', ylabel='mean accuracy (%)',
       title='(c) accuracy against noise SD')
entries = [sum(lists, []) for lists in zip(*(a.get_legend_handles_labels() for a in (axes[0], axes[2])))]
fig.legend(*entries, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.17))
save(fig, 'augment')
save_tex('augment', kAug=f'{K:,}', mAug=f'{M_AUG:,}', sdLow=SD['low'], sdHigh=SD['high'], kSweep=K_SWEEP,
         thr=f'{100 * THRESHOLD:.0f}', gammaOver=OVERFIT['gamma'], COver=OVERFIT['C'], **nums)
print(summary.round(4), sweep.round(3).to_string(), nums, sep='\n')
