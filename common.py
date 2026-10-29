"""Seed, dataset generators, paths, plot style and helpers shared by all scripts."""
import json
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

matplotlib.use('Agg')

SEED = 18042001
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(ROOT, 'data'), os.path.join(ROOT, 'out')
FIGS = os.path.join(OUT, 'figs')
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

N = 10_000                                        # points per dataset, half in each class
TEST = 0.3                                        # held-out share of every dataset
LINEAR = dict(center=1.5, noise=0.4)              # clusters 4.2 apart, 0.4 SD each
SPIRAL = dict(K=4 * np.pi, noise=0.1)             # two turns; neighbouring arms 0.5 apart
SD = {'low': 0.05, 'high': 0.25}                  # augmentation noise: a tenth and a half of that 0.5 gap
M_AUG = 1_000                                     # points per augmented dataset
THRESHOLD = 0.7                                   # bonus: performance threshold
REPS = 5                                          # label shuffles per level in parts 5 and 6
JOBS = 4                                          # parallel workers
CV = StratifiedKFold(5, shuffle=True, random_state=SEED)
WORD = {0: 'Zero', 2: 'Two', 5: 'Five', 10: 'Ten', 15: 'Fifteen', 25: 'TwentyFive', 50: 'Fifty', 75: 'SeventyFive',
        90: 'Ninety', 100: 'Hundred'}             # LaTeX macro names cannot hold digits

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
CLASS = [BLUE, ORANGE]
INK, MUTED, TEXT = '#0b0b0b', '#898781', '#52514e'
RAMP = ['#0d366b', '#1c5cab', '#3987e5', '#86b6ef']  # ordered levels, darkest = cleanest

plt.rcParams.update({'font.size': 13, 'axes.labelsize': 13, 'axes.titlesize': 13, 'legend.fontsize': 12,
                     'lines.linewidth': 2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK, 'xtick.color': TEXT, 'ytick.color': TEXT,
                     'legend.frameon': False, 'savefig.facecolor': 'white'})


def stream(*keys):
    """An independent reproducible random stream per use, e.g. stream(4, 1) for part 4, run 1."""
    return np.random.default_rng([SEED, *keys])


def R(t):
    return t / (2 * np.pi)  # the radius grows by 1 per turn


def ANGLE(t):
    return t  # one turn every 2 pi of t, so the angle keeps looping back


def linear(rng, center, noise, n=N):
    """Two isotropic Gaussian clusters centred at -/+(center, center)."""
    y = np.arange(n) % 2
    return rng.normal(0, noise, (n, 2)) + np.outer(2 * y - 1, [center, center]), y


def arm(t, c, r=R, angle=ANGLE):
    """Noise-free polar point (r(t), angle(t)) of class c in Cartesian coordinates; class 1 is half a turn ahead."""
    theta = angle(t) + np.pi * c
    return np.c_[r(t) * np.cos(theta), r(t) * np.sin(theta)]


def spiral(rng, K, noise, n=N, r=R, angle=ANGLE):
    """Two spiral arms sampled at t ~ U[0, K], with Gaussian noise in both coordinates."""
    t = rng.uniform(0, K, n)
    y = np.arange(n) % 2
    return rng.normal(0, noise, (n, 2)) + arm(t, y, r, angle), y


GENERATORS = {'linear': (linear, LINEAR), 'spiral': (spiral, SPIRAL)}
NAMES = list(GENERATORS)


def load(name):
    """Train/test split saved by simulate.py: Xtr, Xte, ytr, yte."""
    d = pd.read_csv(os.path.join(DATA, f'{name}.csv'))
    X, y, test = d[['x1', 'x2']].values, d['y'].values, (d['split'] == 'test').values
    return X[~test], X[test], y[~test], y[test]


def augment(X, y, sd, k, rng):
    """k new datasets of M_AUG points drawn without replacement, each moved by N(0, sd^2) in every coordinate."""
    idx = np.array([rng.choice(len(X), M_AUG, replace=False) for _ in range(k)])
    return X[idx] + rng.normal(0, sd, (k, M_AUG, X.shape[1])), y[idx]


def shuffle_labels(y, p, rng):
    """Permute the labels of a random p% of the points among themselves."""
    y, idx = y.copy(), rng.choice(len(y), round(p / 100 * len(y)), replace=False)
    y[idx] = y[rng.permutation(idx)]
    return y


def crossing(x, acc, log=False):
    """First x at which acc falls below THRESHOLD, interpolated linearly (on log x if log); nan if acc never
    crosses it from above."""
    acc, t = np.asarray(acc), np.log(x) if log else np.asarray(x, float)
    i = int(np.argmax(acc < THRESHOLD))
    if i == 0:
        return np.nan
    w = (acc[i - 1] - THRESHOLD) / (acc[i - 1] - acc[i])
    v = (1 - w) * t[i - 1] + w * t[i]
    return np.exp(v) if log else v


def best_svm(name):
    """The SVM used from part 4 on: part 2's linear SVM for the clusters, part 3's winning kernel for the spiral."""
    if name == 'linear':
        return SVC(kernel='linear', C=1)
    with open(os.path.join(OUT, 'best.json')) as f:
        return SVC(**json.load(f))


def decision(model, X, chunk=1000):
    """model.decision_function(X) computed in numpy: the same values, several times faster for millions of points."""
    if model.kernel == 'linear':
        return X @ model.coef_[0] + model.intercept_[0]
    if model.kernel != 'rbf' or isinstance(model.gamma, str):
        return model.decision_function(X)
    return np.concatenate([rbf_kernel(p, model.support_vectors_, gamma=model.gamma) @ model.dual_coef_[0]
                           for p in np.array_split(X, math.ceil(len(X) / chunk))]) + model.intercept_[0]


def scatter(ax, X, y, s=2, alpha=0.5, pad=0.15):
    """Both classes, with fixed limits so that lines drawn later cannot rescale the axes."""
    for c in (0, 1):
        ax.scatter(*X[y == c, :2].T, s=s, color=CLASS[c], alpha=alpha, lw=0)
        ax.plot([], [], 'o', ms=5, color=CLASS[c], label=f'class {c}')
    lo, hi = X[:, :2].min() - pad, X[:, :2].max() + pad
    ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect='equal', xlabel='$X_1$', ylabel='$X_2$')


def threshold(ax, x=None):
    """The THRESHOLD line in percent, labelled at x when given."""
    ax.axhline(100 * THRESHOLD, color=INK, lw=0.8, ls='--')
    if x is not None:
        ax.text(x, 100 * THRESHOLD + 1, f'{100 * THRESHOLD:.0f}% threshold', fontsize=11, color=TEXT)


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name + '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_tex(name, **numbers):
    """Numbers quoted in the report, as LaTeX macros (e.g. \\accRBF)."""
    with open(os.path.join(OUT, name + '.tex'), 'w') as f:
        f.writelines(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in numbers.items())
