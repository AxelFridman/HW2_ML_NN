"""Part 2: perceptrons against a linear SVM on both datasets."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron

from common import INK, MUTED, NAMES, OUT, SEED, TEXT, best_svm, load, save, save_tex, scatter

RUNS = 20  # perceptrons fitted from different random orders of the training points


def line(ax, w, b, level=0, **kw):
    x = np.array(ax.get_xlim())
    ax.plot(x, (level - b - w[0] * x) / w[1], **kw)


rows = []
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout='constrained')
for i, (ax, name) in enumerate(zip(axes, NAMES)):
    Xtr, Xte, ytr, yte = load(name)
    svm = best_svm('linear').fit(Xtr, ytr)
    perceptrons = [Perceptron(random_state=SEED + s).fit(Xtr, ytr) for s in range(RUNS)]
    rows += [{'dataset': name, 'model': 'perceptron', 'test': m.score(Xte, yte)} for m in perceptrons]
    rows.append({'dataset': name, 'model': 'linear SVM', 'test': svm.score(Xte, yte)})
    scatter(ax, Xtr, ytr, alpha=0.25)
    for j, m in enumerate(perceptrons):
        line(ax, m.coef_[0], m.intercept_[0], color=MUTED, lw=0.8, label='perceptrons' if j == 0 else None)
    w, b = svm.coef_[0], svm.intercept_[0]
    line(ax, w, b, color=INK, label='linear SVM')
    if name == 'linear':
        for level in (-1, 1):
            line(ax, w, b, level, color=INK, lw=0.8, ls='--', label='SVM margin' if level == 1 else None)
        ax.scatter(*svm.support_vectors_.T, s=40, facecolor='none', edgecolor=INK, lw=0.8)
        ax.plot([], [], 'o', ms=6, mfc='none', mec=INK, mew=0.8, label='support vectors')
    ax.set_title(f'({"ab"[i]}) {name}')
fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=6)

res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/perceptron.csv', index=False)
acc = 100 * res.groupby(['model', 'dataset']).test.agg(['mean', 'std']).fillna(0)
ax = axes[2]
for k, (model, color) in enumerate([('perceptron', MUTED), ('linear SVM', INK)]):
    m = acc.loc[model].reindex(NAMES)
    x = np.arange(len(NAMES)) + (k - 0.5) * 0.38
    ax.bar(x, m['mean'], 0.36, yerr=m['std'], color=color, capsize=3)
    for xi, v, e in zip(x, m['mean'], m['std']):
        ax.text(xi, v + e + 2, f'{v:.0f}', ha='center', fontsize=11, color=TEXT)
ax.axhline(50, color=INK, lw=0.8, ls=':')
ax.set(xticks=range(len(NAMES)), xticklabels=NAMES, ylim=(0, 112), ylabel='test accuracy (%)',
       title='(c) test accuracy')
save(fig, 'perceptron_svm')
nums = {'runs': RUNS}
for name in NAMES:
    nums[f'percTest{name.capitalize()}'] = f'{acc.loc[("perceptron", name), "mean"]:.1f}'
    nums[f'svmTest{name.capitalize()}'] = f'{acc.loc[("linear SVM", name), "mean"]:.1f}'
save_tex('perceptron', **nums)
print(acc.round(1))
