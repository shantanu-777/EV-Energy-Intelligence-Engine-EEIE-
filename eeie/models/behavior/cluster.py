"""K-Means clustering of driver behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BEHAVIOR_CLUSTER_FEATURES = [
    "mean_speed",
    "aggressive_events_per_drive_hr",
    "energy_per_100km",
    "mean_soc",
    "soc_in_mid_band_pct",
]

CLUSTER_LABELS = {0: "efficient", 1: "moderate", 2: "aggressive"}


@dataclass
class BehaviorClusterer:
    """3-class KMeans over per-vehicle behavior summaries."""

    n_clusters: int = 3
    random_state: int = 42
    feature_names: list[str] = field(default_factory=lambda: list(BEHAVIOR_CLUSTER_FEATURES))
    scaler: StandardScaler | None = None
    model: KMeans | None = None
    label_for_cluster: dict[int, str] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame) -> dict[str, float]:
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X[self.feature_names])
        self.model = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=self.random_state,
        ).fit(Xs)

        centroid_aggr = self.scaler.inverse_transform(self.model.cluster_centers_)
        centers_df = pd.DataFrame(centroid_aggr, columns=self.feature_names)
        order = centers_df["energy_per_100km"].argsort().tolist()
        self.label_for_cluster = {
            int(order[0]): "efficient",
            int(order[1]): "moderate",
            int(order[-1]): "aggressive",
        }
        inertia = float(self.model.inertia_)
        logger.info("BehaviorClusterer fit n_clusters={} inertia={:.2f}", self.n_clusters, inertia)
        return {"inertia": inertia}

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Clusterer is not fit.")
        Xs = self.scaler.transform(X[self.feature_names])
        clusters = self.model.predict(Xs)
        labels = [self.label_for_cluster.get(int(c), str(c)) for c in clusters]
        return pd.DataFrame({"cluster": clusters, "label": labels})

    def save(self, path: Path) -> None:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Nothing to save.")
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "label_for_cluster": self.label_for_cluster,
                "n_clusters": self.n_clusters,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> BehaviorClusterer:
        import joblib

        payload = joblib.load(path)
        return cls(
            n_clusters=payload["n_clusters"],
            feature_names=payload["feature_names"],
            scaler=payload["scaler"],
            model=payload["model"],
            label_for_cluster=payload["label_for_cluster"],
        )
