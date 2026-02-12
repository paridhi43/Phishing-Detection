from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass
class TabularTextData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    feature_names: List[str]


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")
    return df


def validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def split_train_val(
    df: pd.DataFrame,
    label_col: str,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 2:
        raise ValueError("Need at least 2 rows to split train/validation")

    stratify = df[label_col] if df[label_col].nunique() > 1 else None
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def build_text_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    max_features: int = 5000,
) -> TabularTextData:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents="unicode",
    )
    x_train = vectorizer.fit_transform(train_df[text_col].astype(str)).toarray()
    x_val = vectorizer.transform(val_df[text_col].astype(str)).toarray()

    return TabularTextData(
        x_train=x_train,
        y_train=train_df[label_col].to_numpy(),
        x_val=x_val,
        y_val=val_df[label_col].to_numpy(),
        feature_names=vectorizer.get_feature_names_out().tolist(),
    )


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    return metrics


def encode_nodes(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    src_col: str,
    dst_col: str,
) -> Tuple[Dict[str, int], pd.DataFrame, pd.DataFrame]:
    train_nodes = set(train_df[src_col].astype(str)).union(set(train_df[dst_col].astype(str)))
    val_nodes = set(val_df[src_col].astype(str)).union(set(val_df[dst_col].astype(str)))
    all_nodes = sorted(train_nodes.union(val_nodes))
    node_to_id = {node: idx for idx, node in enumerate(all_nodes)}

    train_copy = train_df.copy()
    val_copy = val_df.copy()

    train_copy["src_idx"] = train_copy[src_col].astype(str).map(node_to_id)
    train_copy["dst_idx"] = train_copy[dst_col].astype(str).map(node_to_id)

    val_copy["src_idx"] = val_copy[src_col].astype(str).map(node_to_id)
    val_copy["dst_idx"] = val_copy[dst_col].astype(str).map(node_to_id)

    return node_to_id, train_copy, val_copy
