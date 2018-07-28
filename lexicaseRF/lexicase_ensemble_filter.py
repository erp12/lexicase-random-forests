from copy import copy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble.forest import RandomForestClassifier

from lexicaseRF.metrics import squared_error_vector
from lexicaseRF.lexicase import epsilon_lexicase_selection


class LexicaseForestClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, initial_forrest_factor=5, n_estimators=10, **kwargs):
        self._initial_forrest_size = n_estimators * initial_forrest_factor
        self._final_forrest_size = n_estimators

        rf_fit_args = copy(kwargs)
        rf_fit_args.update({'n_estimators': self._initial_forrest_size})
        self._rf = RandomForestClassifier(**rf_fit_args)

    def fit(self, X, y):
        self._rf.fit(X, y)

        for t in self._rf.estimators_:
            tree_y_pred = t.predict(X)
            t._error_vector = squared_error_vector(y, tree_y_pred)

        final_estimators = []
        for i in range(self._final_forrest_size):
            final_estimators.append(epsilon_lexicase_selection(self._rf.estimators_))

        self._rf.estimators_ = final_estimators
        self._rf.n_estimators = self._final_forrest_size
        # TODO: Set other self._rf parameters to match correct size so that predict works.

    def predict(self, X, y=None):
        return self._rf.predict(X)
