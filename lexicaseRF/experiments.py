import pandas as pd

from sklearn.model_selection import train_test_split

from pmlb import fetch_data

import matplotlib.pyplot as plt
import seaborn as sns


def lexicase_filtering_experiments(dataset_names, rf_estimator, lex_rf_estimator,
                                   skip_train=False):
    results = {
        'problem': [],
        'method': [],
        'score': []
    }

    if skip_train:
        results = pd.read_csv("./data/results.csv")
    else:
        for dataset in dataset_names:
            print("Starting", dataset)

            X, y = fetch_data(dataset, return_X_y=True, local_cache_dir='./data/')
            train_X, test_X, train_y, test_y = train_test_split(X, y)

            rf = rf_estimator()
            lexRF = lex_rf_estimator()

            rf.fit(train_X, train_y)
            lexRF.fit(train_X, train_y)

            rf_score = rf.score(test_X, test_y)
            lexRF_score = lexRF.score(test_X, test_y)

            results['problem'] = results['problem'] + ([dataset] * 2)
            results['method'] = results['method'] + ['RF', 'LexRF']
            results['score'].append(rf_score)
            results['score'].append(lexRF_score)

        results = pd.DataFrame(results)
        results.to_csv("./data/results.csv", index=False)

    return results


def plot_results(results):
    problems = (
        results
        .groupby("problem")
        .apply(lambda x: x.score.max() - x.score.min())
        .where(lambda x: x > 0.05)
        .dropna()
        .index.values
    )
    viz_data = results[[x in problems for x in results.problem]]

    # plt.figure(figsize=(30, 3), dpi=192)
    sns.set(style="whitegrid")
    sns.barplot(x="problem", y="score", hue="method", data=viz_data)
    plt.xticks(rotation="vertical")
    plt.ylabel('Test Accuracy')
    plt.show()
