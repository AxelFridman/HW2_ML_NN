"""Part 3: RBF against polynomial kernel on the spiral, each tuned by 5-fold cross-validation on the training set."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from common import CLASS, CV, INK, JOBS, OUT, SPIRAL, R, load, save, save_tex, scatter

C = [0.1, 1, 10]
KERNELS = {'RBF': (SVC(kernel='rbf'), {'gamma': [1, 3, 10, 30, 100], 'C': C}),
           'Poly': (SVC(kernel='poly', coef0=1), {'degree': [3, 5, 7, 9, 11, 13], 'C': C})}
TITLE = {'RBF': 'RBF', 'Poly': 'polynomial'}

Xtr, Xte, ytr, yte = load('spiral')
search = {k: GridSearchCV(m, p, cv=CV, n_jobs=JOBS).fit(Xtr, ytr) for k, (m, p) in KERNELS.items()}

lim = (-R(SPIRAL['K']) - 0.6, R(SPIRAL['K']) + 0.6)
xx, yy = np.meshgrid(np.linspace(*lim, 400), np.linspace(*lim, 400))
fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
nums, cms = {}, {}
for i, (name, gs) in enumerate(search.items()):
    pd.DataFrame(gs.cv_results_).to_csv(f'{OUT}/cv_{name}.csv', index=False)
    m = gs.best_estimator_
    score = m.decision_function(Xte)
    pred = (score > 0).astype(int)
    cms[name] = confusion_matrix(yte, pred)
    nums.update({f'cv{name}': f'{100 * gs.best_score_:.1f}', f'acc{name}': f'{100 * np.mean(pred == yte):.1f}',
                 f'fOne{name}': f'{f1_score(yte, pred):.3f}', f'auc{name}': f'{roc_auc_score(yte, score):.3f}',
                 f'C{name}': f'{m.C:g}'})
    f = m.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax = axes[i]
    ax.contourf(xx, yy, f > 0, levels=[-0.5, 0.5, 1.5], colors=CLASS, alpha=0.12)
    scatter(ax, Xte, yte, s=3, alpha=0.6)
    ax.contour(xx, yy, f, levels=[0], colors=INK, linewidths=1)
    ax.set(xlim=lim, ylim=lim, title=f'({"ab"[i]}) {TITLE[name]}: test accuracy {nums[f"acc{name}"]}%')
axes[0].plot([], [], color=INK, lw=1, label='decision boundary')
axes[0].legend(loc='lower left')
save(fig, 'kernels')

fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
vmax = max(cm.max() for cm in cms.values())
for i, (ax, (name, cm)) in enumerate(zip(axes, cms.items())):
    ax.imshow(cm, cmap='Blues', vmin=0, vmax=vmax)
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, v, ha='center', va='center', fontsize=15, color='white' if v > vmax / 2 else INK)
    ax.set(xticks=[0, 1], yticks=[0, 1], xlabel='predicted class', ylabel='true class',
           title=f'({"ab"[i]}) {TITLE[name]}')
    ax.spines[:].set_visible(False)
save(fig, 'confusion')

best = max(search, key=lambda k: search[k].best_score_)
with open(f'{OUT}/best.json', 'w') as fh:
    json.dump(search[best].best_estimator_.get_params(), fh)
nums.update(gammaRBF=f'{search["RBF"].best_params_["gamma"]:g}', degPoly=search['Poly'].best_params_['degree'])
save_tex('kernels', **nums)
print(nums)
