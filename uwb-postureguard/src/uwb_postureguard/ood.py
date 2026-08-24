"""One-Class SVM OOD detection over LightGBM leaf embeddings."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import OneClassSVM

from .config import OODConfig
from .model import PoseGBDTClassifier


class LeafOODDetector:
    def __init__(self, config: OODConfig, random_state: int = 42):
        self.config = config
        self.random_state = random_state
        self.encoder: OneHotEncoder | None = None
        self.reducer: TruncatedSVD | None = None
        self.scaler: StandardScaler | None = None
        self.estimator: OneClassSVM | None = None
        self.threshold_: float | None = None

    def _sample(self, X: pd.DataFrame) -> pd.DataFrame:
        if len(X) <= self.config.max_train_samples:
            return X
        rng = np.random.default_rng(self.random_state)
        indices = np.sort(rng.choice(len(X), size=self.config.max_train_samples, replace=False))
        return X.iloc[indices]

    def _fit_embedding(self, leaves: np.ndarray) -> np.ndarray:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
        one_hot = encoder.fit_transform(leaves)
        max_components = min(one_hot.shape[0] - 1, one_hot.shape[1] - 1)
        component_count = min(self.config.svd_components, max_components)
        has_embedding_variation = np.unique(leaves, axis=0).shape[0] > 1
        if component_count >= 2 and has_embedding_variation:
            reducer = TruncatedSVD(n_components=component_count, random_state=self.random_state)
            embedded = reducer.fit_transform(one_hot)
            self.reducer = reducer
        else:
            embedded = one_hot.toarray()
            self.reducer = None
        scaler = StandardScaler()
        scaled = scaler.fit_transform(embedded)
        self.encoder = encoder
        self.scaler = scaler
        return scaled

    def _transform_leaves(self, leaves: np.ndarray) -> np.ndarray:
        if self.encoder is None or self.scaler is None:
            raise RuntimeError("LeafOODDetector has not been fitted")
        one_hot = self.encoder.transform(leaves)
        embedded = self.reducer.transform(one_hot) if self.reducer else one_hot.toarray()
        return self.scaler.transform(embedded)

    def fit(self, classifier: PoseGBDTClassifier, X: pd.DataFrame) -> LeafOODDetector:
        self.config.validate()
        sampled = self._sample(X)
        leaves = classifier.predict_leaves(sampled)
        embedded = self._fit_embedding(leaves)
        estimator = OneClassSVM(kernel="rbf", gamma="scale", nu=self.config.nu)
        estimator.fit(embedded)
        scores = estimator.decision_function(embedded).reshape(-1)
        self.estimator = estimator
        self.threshold_ = float(np.quantile(scores, self.config.score_quantile))
        return self

    def score_samples(self, classifier: PoseGBDTClassifier, X: pd.DataFrame) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("LeafOODDetector has not been fitted")
        embedded = self._transform_leaves(classifier.predict_leaves(X))
        return self.estimator.decision_function(embedded).reshape(-1)

    def is_ood(self, classifier: PoseGBDTClassifier, X: pd.DataFrame) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("LeafOODDetector has not been fitted")
        return self.score_samples(classifier, X) < self.threshold_
