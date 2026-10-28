"""Part 3: RBF against polynomial kernel on the spiral, each tuned by 5-fold cross-validation on the training set."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from common import AQUA, CLASS, CV, INK, JOBS, MUTED, OUT, SPIRAL, YELLOW, R, load, save, save_tex, scatter

C = [0.1, 1, 10]
KERNELS = {'RBF': (SVC(kernel='rbf'), {'gamma': [1, 3, 10, 30, 100], 'C': C}),
           'Poly': (SVC(kernel='poly', coef0=1), {'degree': [3, 5, 7, 9, 11, 13], 'C': C})}
COLOR = {'RBF': AQUA, 'Poly': YELLOW}
TITLE = {'RBF': 'RBF', 'Poly': 'polynomial'}

Xtr, Xte, ytr, yte = load('spiral')
search = {k: GridSearchCV(m, p, cv=CV, n_jobs=JOBS).fit(Xtr, ytr) for k, (m, p) in KERNELS.items()}
results = {k: pd.DataFrame(gs.cv_results_) for k, gs in search.items()}

lim = (-R(SPIRAL['K']) - 0.6, R(SPIRAL['K']) + 0.6)
xx, yy = np.meshgrid(np.linspace(*lim, 400), np.linspace(*lim, 400))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
nums = {}
for i, (name, gs) in enumerate(search.items()):
    results[name].to_csv(f'{OUT}/cv_{name}.csv', index=False)
    m = gs.best_estimator_
    score = m.decision_function(Xte)
    pred = (score > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    nums.update({f'cv{name}': f'{100 * gs.best_score_:.1f}', f'acc{name}': f'{100 * np.mean(pred == yte):.1f}',
                 f'fOne{name}': f'{f1_score(yte, pred):.3f}', f'auc{name}': f'{roc_auc_score(yte, score):.3f}',
                 f'tn{name}': tn, f'fp{name}': fp, f'fn{name}': fn, f'tp{name}': tp, f'C{name}': f'{m.C:g}'})
    f = m.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax = axes[i]
    ax.contourf(xx, yy, f > 0, levels=[-0.5, 0.5, 1.5], colors=CLASS, alpha=0.12)
    scatter(ax, Xte, yte, s=3, alpha=0.6)
    ax.contour(xx, yy, f, levels=[0], colors=INK, linewidths=1)
    ax.set(xlim=lim, ylim=lim, title=f'({"ab"[i]}) {TITLE[name]}: test accuracy {nums[f"acc{name}"]}%')
    fpr, tpr, _ = roc_curve(yte, score)
    axes[2].plot(fpr, tpr, color=COLOR[name], label=f'{TITLE[name]} (AUC {nums[f"auc{name}"]})')
axes[0].plot([], [], color=INK, lw=1, label='decision boundary')
axes[0].legend(loc='lower left', fontsize=9)
axes[2].plot([0, 1], [0, 1], color=MUTED, lw=0.8)
axes[2].set(xlabel='false-positive rate', ylabel='true-positive rate', aspect='equal', title='(c) ROC on the test set')
axes[2].legend(loc='lower right')
save(fig, 'kernels')

best = max(search, key=lambda k: search[k].best_score_)
with open(f'{OUT}/best.json', 'w') as fh:
    json.dump(search[best].best_estimator_.get_params(), fh)
rbf, gamma = results['RBF'], search['RBF'].best_params_['gamma']
by_degree = results['Poly'].groupby('param_degree').mean_test_score.max()
by_gamma = rbf.groupby('param_gamma').mean_test_score.max()
by_C = rbf[rbf.param_gamma == gamma].set_index('param_C').mean_test_score.sort_index()
fit_time = {k: gs.cv_results_['mean_fit_time'][gs.best_index_] for k, gs in search.items()}
nums.update(gammaRBF=f'{gamma:g}', degPoly=search['Poly'].best_params_['degree'], degLow=by_degree.index[0],
            cvPolyLow=f'{100 * by_degree.iloc[0]:.0f}', cvPolyPrev=f'{100 * by_degree.iloc[-2]:.2f}',
            cvPolyTop=f'{100 * by_degree.iloc[-1]:.2f}', degPrev=by_degree.index[-2],
            cvRBFlo=f'{100 * by_gamma[by_gamma.index >= gamma].min():.1f}',
            cvGainC=f'{100 * (by_C.iloc[-1] - by_C.iloc[-2]):.2f}',
            cvSD=f'{100 * rbf.std_test_score[search["RBF"].best_index_]:.2f}',
            fitRBF=f'{fit_time["RBF"]:.1f}', fitPoly=f'{fit_time["Poly"]:.0f}')
save_tex('kernels', **nums)
print(by_degree.round(4), by_gamma.round(4), by_C.round(4), nums, sep='\n')
