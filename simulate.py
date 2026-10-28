"""Part 1: simulate the linear and spiral datasets, split them 70/30, save them to data/ and plot them."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import DATA, GENERATORS, INK, LINEAR, N, SEED, SPIRAL, TEST, arm, bayes, save, save_tex, scatter, stream

nums = {}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
for i, (ax, (name, (make, params))) in enumerate(zip(axes, GENERATORS.items())):
    X, y = make(rng=stream(1, i), **params)
    test = np.zeros(N, bool)
    test[train_test_split(np.arange(N), test_size=TEST, stratify=y, random_state=SEED)[1]] = True
    pd.DataFrame({'x1': X[:, 0], 'x2': X[:, 1], 'y': y, 'split': np.where(test, 'test', 'train')}).to_csv(
        f'{DATA}/{name}.csv', index=False)
    nums[f'bayesTest{name}'] = f'{100 * np.mean((bayes(name, X[test], params["noise"]) > 0) == y[test]):.1f}'
    scatter(ax, X, y)
    ax.set_title(f'({"ab"[i]}) {name}')
for c in (0, 1):
    axes[1].plot(*arm(np.linspace(0, SPIRAL['K'], 600), c).T, color=INK, lw=0.8,
                 label='noise-free arm' if c == 0 else None)
axes[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
save(fig, 'datasets')

save_tex('simulate', nAll=f'{N:,}', nTest=f'{test.sum():,}', nTrain=f'{(~test).sum():,}',
         centerLin=LINEAR['center'], noiseLin=LINEAR['noise'], noiseSpi=SPIRAL['noise'],
         Kspi=f'{SPIRAL["K"] / np.pi:g}\\pi', turns=f'{SPIRAL["K"] / (2 * np.pi):g}',
         gapLin=f'{2 * np.sqrt(2) * LINEAR["center"]:.1f}', **nums)
