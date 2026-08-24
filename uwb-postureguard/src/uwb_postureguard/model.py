"""LightGBM backbone for the PoseGBDT classifier."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import ModelConfig


class PoseGBDTClassifier:
    def __init__(self, config: ModelConfig, random_state: int = 42):
        self.config = config
        self.random_state = random_state
        self.estimator: lgb.LGBMClassifier | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
    ) -> PoseGBDTClassifier:
        self.config.validate()
        estimator = lgb.LGBMClassifier(
            objective="multiclass",
            boosting_type="gbdt",
            num_leaves=self.config.num_leaves,
            learning_rate=self.config.learning_rate,
            feature_fraction=self.config.feature_fraction,
            n_estimators=self.config.max_rounds,
            n_jobs=self.config.n_jobs,
            random_state=self.random_state,
            force_col_wise=True,
            verbosity=-1,
        )
        estimator.fit(
            X_train,
            y_train,
            eval_X=X_validation,
            eval_y=y_validation,
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(self.config.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        self.estimator = estimator
        return self

    def _require_estimator(self) -> lgb.LGBMClassifier:
        if self.estimator is None:
            raise RuntimeError("PoseGBDTClassifier has not been fitted")
        return self.estimator

    @property
    def classes_(self) -> np.ndarray:
        return self._require_estimator().classes_

    @property
    def best_iteration_(self) -> int:
        estimator = self._require_estimator()
        return int(estimator.best_iteration_ or estimator.n_estimators_)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._require_estimator().predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._require_estimator().predict_proba(X)

    def predict_leaves(self, X: pd.DataFrame) -> np.ndarray:
        estimator = self._require_estimator()
        leaves = estimator.booster_.predict(X, pred_leaf=True, num_iteration=self.best_iteration_)
        leaves = np.asarray(leaves)
        return leaves.reshape(-1, 1) if leaves.ndim == 1 else leaves
