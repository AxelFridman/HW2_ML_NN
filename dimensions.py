"""Part 6: part 5's label shuffling after adding d - 2 pure-noise coordinates to both datasets (d = 5, 10, 15)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from common import (CV, GENERATORS, JOBS, NAMES, OUT, RAMP, REPS, WORD, best_svm, decision, load, save, save_tex,
                    shuffle_labels, stream)

DIMS = [2, 5, 10, 15]              # d = 2 repeats part 5: the same points, model and shuffles
P = [0, 50, 75, 90]                # no shuffling, the largest level asked for, and where part 5 starts to fail
GRID = {'gamma': [0.3, 1, 3, 10, 30, 100], 'C': [0.01, 0.1, 1, 10]}


def dataset(name, dim):
    """Part 1's points and split, with dim - 2 extra coordinates of pure noise at the dataset's noise SD."""
    Xtr, Xte, ytr, yte = load(name)
    rng, noise = stream(6, NAMES.index(name), dim), GENERATORS[name][1]['noise']
    Xtr, Xte = (np.c_[X, rng.normal(0, noise, (len(X), dim - 2))] for X in (Xtr, Xte))
    return Xtr, Xte, ytr, yte


def run(name, dim, p, rep, model, data):
    Xtr, Xte, ytr, yte = data
    noisy = shuffle_labels(ytr, p, stream(5, NAMES.index(name), p, rep))  # the same wrong labels as in part 5
    model.fit(Xtr, noisy)
    wrong, fitted = noisy != ytr, (decision(model, Xtr) > 0) == noisy
    return {'dataset': name, 'dim': dim, 'p': p, 'rep': rep, 'clean': model.score(Xte, yte),
            'memorised': fitted[wrong].mean() if wrong.any() else np.nan}


jobs, nums = [], {}
for name in NAMES:
    for dim in DIMS:
        data = dataset(name, dim)
        if name == 'linear' or dim == 2:
            model = best_svm(name)
        else:
            gs = GridSearchCV(SVC(kernel='rbf'), GRID, cv=CV, n_jobs=JOBS).fit(data[0], data[2])
            pd.DataFrame(gs.cv_results_).to_csv(f'{OUT}/cv_spiral_{dim}d.csv', index=False)
            model = SVC(kernel='rbf', **gs.best_params_)
            nums.update({f'gamma{WORD[dim]}': f'{gs.best_params_["gamma"]:g}',
                         f'C{WORD[dim]}': f'{gs.best_params_["C"]:g}', f'cv{WORD[dim]}': f'{100 * gs.best_score_:.1f}'})
            print(dim, gs.best_params_, round(gs.best_score_, 4), flush=True)
        jobs += [delayed(run)(name, dim, p, rep, model, data) for p in P for rep in range(REPS)]
res = pd.DataFrame(Parallel(n_jobs=JOBS, verbose=5)(jobs))
res.to_csv(f'{OUT}/dimensions.csv', index=False)
mean, std = (100 * res.groupby(['dataset', 'p', 'dim']).clean.agg(a) for a in ['mean', 'std'])
memorised = 100 * res.groupby(['dataset', 'p', 'dim']).memorised.mean()
nums.update({f'memo{WORD[d]}': f'{memorised["spiral", 75, d]:.0f}' for d in DIMS})

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
for i, (ax, name) in enumerate(zip(axes, NAMES)):
    for p, color in zip(P, RAMP):
        m, s = mean.loc[name, p], std.loc[name, p]
        ax.plot(DIMS, m, 'o-', ms=4, color=color, label=f'{p}% shuffled')
        ax.fill_between(DIMS, m - s, np.minimum(m + s, 100), color=color, alpha=0.15, lw=0)
        nums.update({f'dim{name}{WORD[p]}{WORD[d]}': f'{m[d]:.1f}' for d in DIMS})
    ax.set(xticks=DIMS, xlabel='dimension $d$', title=f'({"ab"[i]}) {name}, best SVM')
axes[0].set(ylabel='test accuracy (%)')
axes[0].legend(loc='lower left')
save(fig, 'dimensions')
save_tex('dimensions', **nums)
print(mean.round(2), std.round(2), memorised.round(1), nums, sep='\n')
