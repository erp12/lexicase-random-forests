import argparse

from sklearn.ensemble.forest import (
    RandomForestClassifier,
    RandomForestRegressor
)

from lexicaseRF.lexicase_ensemble_filter import (
    LexicaseForestClassifier,
    LexicaseForestRegressor
)
from lexicaseRF.experiments import (
    lexicase_filtering_experiments,
    plot_results
)

from pmlb import (
    classification_dataset_names,
    regression_dataset_names
)


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'method',
        nargs='?',
        choices=[
            'filter',
            'ga'
        ])
    parser.add_argument(
        'task',
        nargs='?',
        choices=[
            'classification',
            'regression'
        ])
    parser.add_argument('--skip_train', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':

    cli_args = get_cli_args()

    if cli_args.method == "filter":
        if cli_args.task == "classification":
            results = lexicase_filtering_experiments(classification_dataset_names,
                                                     RandomForestClassifier,
                                                     LexicaseForestClassifier,
                                                     cli_args.skip_train)
            plot_results(results)
        elif cli_args.task == "regression":
            results = lexicase_filtering_experiments(regression_dataset_names,
                                                     RandomForestRegressor,
                                                     LexicaseForestRegressor,
                                                     cli_args.skip_train)
            plot_results(results)
    elif cli_args.method == "ga":
        print("Not implemented yet.")
