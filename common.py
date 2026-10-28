"""Seed, dataset generators, paths, plot style and helpers shared by all scripts."""
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
CLASS = [BLUE, ORANGE]
INK, MUTED, TEXT = '#0b0b0b', '#898781', '#52514e'
RAMP = ['#0d366b', '#1c5cab', '#3987e5', '#86b6ef']  # ordered levels, darkest = cleanest

plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12, 'legend.fontsize': 10,
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


def best_svm(name):
    """The SVM used from part 4 on: part 2's linear SVM for the clusters, part 3's winning kernel for the spiral."""
    if name == 'linear':
        return SVC(kernel='linear', C=1)
    with open(os.path.join(OUT, 'best.json')) as f:
        return SVC(**json.load(f))


def scatter(ax, X, y, s=2, alpha=0.5, pad=0.15):
    """Both classes, with fixed limits so that lines drawn later cannot rescale the axes."""
    for c in (0, 1):
        ax.scatter(*X[y == c, :2].T, s=s, color=CLASS[c], alpha=alpha, lw=0)
        ax.plot([], [], 'o', ms=5, color=CLASS[c], label=f'class {c}')
    lo, hi = X[:, :2].min() - pad, X[:, :2].max() + pad
    ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect='equal', xlabel='$X_1$', ylabel='$X_2$')


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name + '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_tex(name, **numbers):
    """Numbers quoted in the report, as LaTeX macros (e.g. \\accRBF)."""
    with open(os.path.join(OUT, name + '.tex'), 'w') as f:
        f.writelines(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in numbers.items())
