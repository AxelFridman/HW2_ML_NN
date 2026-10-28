# HW2: perceptron, SVM and kernels on simulated data

## Run

```
pip install -r requirements.txt
python make_all.py               # every part in order, about 1.5 hours with 4 workers
python make_all.py kernels.py    # or only the scripts named
```

`hw2.ipynb` runs the same scripts, one cell per part, and shows their figures (it needs Jupyter).
`simulate.py` writes the two datasets to `data/`; every figure and every computed number of the report goes to `out/`,
which is not committed. All randomness comes from the seed `18042001` in `common.py`, which also holds the dataset
parameters (size, noise, number of spiral turns, `r(t)` and `angle(t)`) and the number of parallel workers (`JOBS`).
The report's numbers come from Python 3.8.10 with numpy 1.24.4, scipy 1.10.1, scikit-learn 1.3.2, pandas 2.0.3,
matplotlib 3.7.5 and joblib 1.4.2.

| File | What it does |
|---|---|
| `common.py` | Seed, generators and their parameters, data loader, augmentation, label shuffling, the Bayes classifier, 70% crossing, best SVM, a fast numpy SVM decision function, plot style, LaTeX macros |
| `simulate.py` | Part 1: both datasets (n = 10,000), a stratified 70/30 split, `data/linear.csv` and `data/spiral.csv`, scatter plots |
| `perceptron.py` | Part 2: 20 perceptrons against a linear SVM on both datasets |
| `kernels.py` | Part 3: RBF against polynomial kernel on the spiral, tuned by 5-fold CV; the winner goes to `out/best.json` |
| `augment.py` | Part 4 and bonus: k = 10,000 augmented datasets at a low and a high noise SD, and accuracy against the SD |
| `shuffle.py` | Part 5 and bonus: accuracy against the share of shuffled training labels |
| `dimensions.py` | Part 6: part 5's shuffles at 0, 50, 75 and 90% after adding 3, 8 or 13 seeded noise coordinates, with the spiral's RBF retuned by CV per dimension; test accuracy against the dimension |
| `make_all.py` | Runs the scripts in order |
| `hw2.ipynb` | Every part end to end in a notebook, with the figures |
