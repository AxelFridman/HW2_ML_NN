"""Part 2: perceptrons against a linear SVM on both datasets."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC

from common import INK, MUTED, NAMES, OUT, SEED, best_svm, load, save, save_tex, scatter

RUNS = 20  # perceptrons fitted from different random orders of the training points


def margin(w, b, X, y):
    """Distance from the boundary to the closest point (negative when a point is on the wrong side)."""
    return np.min((2 * y - 1) * (X @ w + b)) / np.linalg.norm(w)


def line(ax, w, b, level=0, **kw):
    x = np.array(ax.get_xlim())
    ax.plot(x, (level - b - w[0] * x) / w[1], **kw)


rows, nums = [], {}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
for ax, name in zip(axes, NAMES):
    Xtr, Xte, ytr, yte = load(name)
    svm = best_svm('linear').fit(Xtr, ytr)
    models = [('Perceptron', Perceptron(random_state=SEED + s).fit(Xtr, ytr)) for s in range(RUNS)]
    models.append(('Linear SVM', svm))
    for label, m in models:
        rows.append({'dataset': name, 'model': label, 'test': m.score(Xte, yte),
                     'margin': margin(m.coef_[0], m.intercept_[0], Xtr, ytr)})
    scatter(ax, Xtr, ytr, alpha=0.25)
    for j, (_, m) in enumerate(models[:-1]):
        line(ax, m.coef_[0], m.intercept_[0], color=MUTED, lw=0.8, label='perceptrons' if j == 0 else None)
    w, b = svm.coef_[0], svm.intercept_[0]
    line(ax, w, b, color=INK, label='linear SVM')
    if name == 'linear':
        for level in (-1, 1):
            line(ax, w, b, level, color=INK, lw=0.8, ls='--', label='SVM margin' if level == 1 else None)
        ax.scatter(*svm.support_vectors_.T, s=40, facecolor='none', edgecolor=INK, lw=0.8)
        ax.plot([], [], 'o', ms=6, mfc='none', mec=INK, mew=0.8, label='support vectors')
        hard = SVC(kernel='linear', C=1e4).fit(Xtr, ytr)  # C so large that no point may enter the margin
        nums.update(nSV=len(svm.support_), hardMargin=f'{margin(hard.coef_[0], hard.intercept_[0], Xtr, ytr):.2f}')
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.06))

res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/perceptron.csv', index=False)
for i, (ax, (name, d)) in enumerate(zip(axes, res.groupby('dataset'))):
    p, s = d[d.model == 'Perceptron'], d[d.model == 'Linear SVM'].iloc[0]
    tag = name.capitalize()
    nums.update({f'percTest{tag}': f'{100 * p.test.mean():.1f}', f'percTestSD{tag}': f'{100 * p.test.std():.1f}',
                 f'percTestMax{tag}': f'{100 * p.test.max():.1f}', f'svmTest{tag}': f'{100 * s.test:.1f}'})
    ax.set_title(f'({"ab"[i]}) {name}: perceptrons {nums[f"percTest{tag}"]} $\\pm$ {nums[f"percTestSD{tag}"]}%, '
                 f'SVM {nums[f"svmTest{tag}"]}%')
save(fig, 'perceptron_svm')
lin = res[res.dataset == 'linear']
p = lin[lin.model == 'Perceptron'].margin
nums.update(svmMargin=f'{lin[lin.model == "Linear SVM"].margin.iloc[0]:.2f}', percMarginMed=f'{p.median():.2f}',
            percMarginMin=f'{p.min():.2f}', percMarginMax=f'{p.max():.2f}', runs=RUNS)
save_tex('perceptron', **nums)
print(res.groupby(['dataset', 'model'])[['test', 'margin']].describe().T.round(3))
