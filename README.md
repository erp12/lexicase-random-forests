# Lexicase Selection and Random Forests

## How to run experiments.

1. Create and activate a virtual environment for python 3.
2. Install requirements.
3. Run the main `LexicaseRF.py` script

```
python -m venv venv/
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python lexicaseRF.py
```

A `results.csv` file will be generated in the `data/` folder and a plot will be shown.

To redraw the

```
python lexicaseRF.py --skip-train
```

## ToDo:

- [ ] Grid search n_estimators, depth, and initial_forrest_factor
- [ ] Add LexicaseForestRegressor and regression benchmarks
- [ ] [Other related experiments mentioned on the push-language discourse](https://push-language.hampshire.edu/t/lexicase-tree-bagging/1185/6?u=erp12)
