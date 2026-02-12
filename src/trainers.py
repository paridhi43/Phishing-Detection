from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict

import lightgbm as lgb
import numpy as np

from src.data_utils import binary_metrics, build_text_features, encode_nodes


@dataclass
class TrainConfig:
    output_dir: str
    text_col: str
    label_col: str
    epochs: int = 3
    batch_size: int = 16
    lr: float = 2e-5
    max_length: int = 128
    bert_model: str = "distilbert-base-uncased"


def _save_metrics(metrics: Dict[str, float], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def train_lightgbm(train_df, val_df, cfg: TrainConfig) -> Dict[str, float]:
    data = build_text_features(
        train_df,
        val_df,
        text_col=cfg.text_col,
        label_col=cfg.label_col,
    )

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary",
        class_weight="balanced",
        random_state=42,
    )
    model.fit(data.x_train, data.y_train)
    probs = model.predict_proba(data.x_val)[:, 1]
    metrics = binary_metrics(data.y_val, probs)

    model_dir = os.path.join(cfg.output_dir, "lgbm")
    os.makedirs(model_dir, exist_ok=True)
    model.booster_.save_model(os.path.join(model_dir, "lightgbm_model.txt"))
    _save_metrics(metrics, model_dir)
    return metrics


def train_bert(train_df, val_df, cfg: TrainConfig) -> Dict[str, float]:
    try:
        import torch
        import torch.nn as nn
        from sklearn.utils.class_weight import compute_class_weight
        from torch.utils.data import DataLoader, Dataset
        from tqdm.auto import tqdm
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError("BERT training requires torch, transformers, sklearn, and tqdm installed") from exc

    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length: int) -> None:
            self.texts = texts.astype(str).tolist()
            self.labels = labels.astype(int).tolist()
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int):
            encoded = self.tokenizer(
                self.texts[idx], max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    class BERTClassifier(nn.Module):
        def __init__(self, model_name: str, dropout: float = 0.2) -> None:
            super().__init__()
            self.bert = AutoModel.from_pretrained(model_name)
            hidden = self.bert.config.hidden_size
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden, 2)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            return self.fc(self.dropout(outputs.last_hidden_state[:, 0]))

    tokenizer = AutoTokenizer.from_pretrained(cfg.bert_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(TextDataset(train_df[cfg.text_col], train_df[cfg.label_col], tokenizer, cfg.max_length), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TextDataset(val_df[cfg.text_col], val_df[cfg.label_col], tokenizer, cfg.max_length), batch_size=cfg.batch_size, shuffle=False)

    model = BERTClassifier(cfg.bert_model).to(device)
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_df[cfg.label_col].to_numpy())
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    model.train()
    for _ in range(cfg.epochs):
        for batch in tqdm(train_loader, desc="BERT training", leave=False):
            optimizer.zero_grad()
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()

    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="BERT validation", leave=False):
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            probs_all.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist())
            labels_all.extend(batch["labels"].cpu().numpy().tolist())

    metrics = binary_metrics(np.array(labels_all), np.array(probs_all))
    model_dir = os.path.join(cfg.output_dir, "bert")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "model_bert.pth"))
    tokenizer.save_pretrained(os.path.join(model_dir, "tokenizer"))
    _save_metrics(metrics, model_dir)
    return metrics


def train_gnn(train_df, val_df, cfg: TrainConfig, src_col: str, dst_col: str) -> Dict[str, float]:
    try:
        import torch
        import torch.nn as nn
        from sklearn.feature_extraction.text import HashingVectorizer
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv
    except ImportError as exc:
        raise ImportError("GNN training requires torch and torch-geometric installed") from exc

    class PhishingGCN(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.drop = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_dim, 1)

        def forward(self, x, edge_index):
            x = self.drop(self.conv1(x, edge_index).relu())
            x = self.drop(self.conv2(x, edge_index).relu())
            return self.classifier(x).squeeze(-1)

    req = [cfg.text_col, cfg.label_col, src_col, dst_col]
    train = train_df[req].copy()
    val = val_df[req].copy()
    _, train_graph, val_graph = encode_nodes(train, val, src_col, dst_col)

    all_text = np.concatenate([train[cfg.text_col].astype(str).to_numpy(), val[cfg.text_col].astype(str).to_numpy()])
    text_feats = HashingVectorizer(n_features=256, alternate_sign=False).fit_transform(all_text).toarray()

    n_nodes = int(max(train_graph[["src_idx", "dst_idx"]].to_numpy().max(), val_graph[["src_idx", "dst_idx"]].to_numpy().max()) + 1)
    x = np.zeros((n_nodes, text_feats.shape[1]), dtype=np.float32)

    for i, row in enumerate(train_graph.itertuples(index=False)):
        x[int(row.src_idx)] += text_feats[i]
        x[int(row.dst_idx)] += text_feats[i]
    offset = len(train_graph)
    for j, row in enumerate(val_graph.itertuples(index=False)):
        x[int(row.src_idx)] += text_feats[offset + j]
        x[int(row.dst_idx)] += text_feats[offset + j]

    edge_index_train = torch.tensor(train_graph[["src_idx", "dst_idx"]].to_numpy().T, dtype=torch.long)
    edge_index_val = torch.tensor(val_graph[["src_idx", "dst_idx"]].to_numpy().T, dtype=torch.long)
    y_train = train_graph[cfg.label_col].to_numpy().astype(np.float32)
    y_val = val_graph[cfg.label_col].to_numpy().astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhishingGCN(in_dim=x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    graph = Data(x=torch.tensor(x, dtype=torch.float32), edge_index=edge_index_train).to(device)
    train_nodes = torch.tensor(train_graph["src_idx"].to_numpy(), dtype=torch.long, device=device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    model.train()
    for _ in range(max(cfg.epochs, 20)):
        optimizer.zero_grad()
        loss = criterion(model(graph.x, graph.edge_index)[train_nodes], y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        full_edges = torch.cat([edge_index_train, edge_index_val], dim=1).to(device)
        logits = model(torch.tensor(x, dtype=torch.float32, device=device), full_edges)
        probs = torch.sigmoid(logits[torch.tensor(val_graph["src_idx"].to_numpy(), dtype=torch.long, device=device)]).cpu().numpy()

    metrics = binary_metrics(y_val, probs)
    model_dir = os.path.join(cfg.output_dir, "gnn")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "gnn_model.pt"))
    _save_metrics(metrics, model_dir)
    return metrics
