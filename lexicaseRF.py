import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble.forest import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

from lexicaseRF.lexicase_ensemble_filter import (
    LexicaseForestClassifier,
)

from pmlb import (
    fetch_data,
    classification_dataset_names,
)


if __name__ == '__main__':

    results = {
        'problem': [],
        'method': [],
        'score': []
    }

    if len(sys.argv) > 1 and sys.argv[1] == '--skip-train':
        results = pd.read_csv("./data/results.csv")
    else:
        for classification_dataset in classification_dataset_names:
            print("Starting", classification_dataset)

            X, y = fetch_data(classification_dataset, return_X_y=True, local_cache_dir='./data/')
            train_X, test_X, train_y, test_y = train_test_split(X, y)

            rf = RandomForestClassifier()
            lexRF = LexicaseForestClassifier()

            rf.fit(train_X, train_y)
            lexRF.fit(train_X, train_y)

            rf_score = rf.score(test_X, test_y)
            lexRF_score = lexRF.score(test_X, test_y)

            results['problem'] = results['problem'] + ([classification_dataset] * 2)
            results['method'] = results['method'] + ['RF', 'LexRF']
            results['score'].append(rf_score)
            results['score'].append(lexRF_score)

        results = pd.DataFrame(results)
        results.to_csv("./data/results.csv", index=False)

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
    g = sns.barplot(x="problem", y="score", hue="method", data=viz_data)
    plt.xticks(rotation="vertical")
    plt.ylabel('Test Accuracy')

    plt.show()
