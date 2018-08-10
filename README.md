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
python lexicaseRF.py [method] [task]
```

### Methods

Currently there is only one supported method to experiment with: `filter`
These experiments are for using lexicsae selections to pick the trees in a
random forest.

### Tasks

The supported tasks are `classification` and `regression`.

### Caching training data and results

Training data and results will be stored in the `data/` folder for faster
reading during subsequent runs.

A `results.csv` file will be generated in the `data/` folder which can be used
to analyze results without rerunning experiments. To use the `results.csv`
from previous experiements, add the `--skip-train` flag when running the script.

```
python lexicaseRF.py --skip-train
```

## ToDo:

- [ ] Prevent lexicase from selecting the same tree twice?
- [ ] Grid search n_estimators, depth, and overpopulate_multiplier.
- [ ] [Other related experiments mentioned on the push-language discourse](https://push-language.hampshire.edu/t/lexicase-tree-bagging/1185/6?u=erp12)
