"""Part 5 and bonus: shuffle a share of the training labels, refit the best SVM, score clean and augmented data."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import (JOBS, NAMES, OUT, RAMP, REPS, SD, THRESHOLD, WORD, augment, best_svm, crossing, decision, load,
                    save, save_tex, shuffle_labels, stream, threshold)

P = [0, 15, 25, 50, 75, 90, 100]   # % of training labels shuffled: the three asked for, then on to the 70% point
K_EVAL = 100                       # augmented datasets per SD for every refitted model: 100,000 points
COLOR = {'clean': RAMP[0], 'low': RAMP[2], 'high': RAMP[3]}


def run(name, p, rep, aug):
    Xtr, Xte, ytr, yte = load(name)
    noisy = shuffle_labels(ytr, p, stream(5, NAMES.index(name), p, rep))
    m = best_svm(name).fit(Xtr, noisy)
    return {'dataset': name, 'p': p, 'rep': rep, 'wrong': np.mean(noisy != ytr), 'clean': m.score(Xte, yte),
            **{level: np.mean((decision(m, Xa.reshape(-1, Xa.shape[-1])) > 0) == ya.ravel())
               for level, (Xa, ya) in aug.items()}}


aug = {}
for d, name in enumerate(NAMES):
    Xtr, _, ytr, _ = load(name)
    aug[name] = {level: augment(Xtr, ytr, sd, K_EVAL, stream(5, d, j)) for j, (level, sd) in enumerate(SD.items())}
res = pd.DataFrame(Parallel(n_jobs=JOBS, verbose=5)(
    delayed(run)(name, p, rep, aug[name]) for name in NAMES for p in P for rep in range(REPS)))
res.to_csv(f'{OUT}/shuffle.csv', index=False)
mean, std = res.groupby(['dataset', 'p']).mean(), res.groupby(['dataset', 'p']).std()

nums = {}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
for i, (ax, name) in enumerate(zip(axes, NAMES)):
    for level, color in COLOR.items():
        m, s = 100 * mean.loc[name, level], 100 * std.loc[name, level]
        label = 'clean test set' if level == 'clean' else f'{level}-SD augmentation ($\\sigma$ = {SD[level]})'
        ax.plot(P, m, 'o-', ms=4, color=color, label=label)
        ax.fill_between(P, m - s, np.minimum(m + s, 100), color=color, alpha=0.15, lw=0)
        star = crossing(P, m / 100)
        nums[f'pStar{name}{level}'] = '--' if np.isnan(star) else f'{star:.0f}'
        if not np.isnan(star):
            ax.plot(star, 100 * THRESHOLD, 'o', ms=7, mfc='white', mec=color, mew=2)
        nums.update({f'acc{name}{level}{WORD[p]}': f'{m[p]:.1f}' for p in P})
    threshold(ax, 1 if i == 0 else None)
    ax.set(xticks=P, xlabel='training labels shuffled (%)', title=f'({"ab"[i]}) {name}, best SVM')
axes[0].set(ylabel='accuracy (%)')
axes[0].legend(loc='lower left', fontsize=9)
save(fig, 'shuffle')
wrong = 100 * res.groupby('p').wrong.mean()
save_tex('shuffle', reps=REPS, kEval=K_EVAL, wrongFifty=f'{wrong[50]:.1f}', wrongHundred=f'{wrong[100]:.1f}', **nums)
print(mean.round(4), std.round(4), nums, sep='\n')
