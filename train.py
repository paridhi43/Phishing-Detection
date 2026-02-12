from __future__ import annotations

import argparse
import json
import os

from src.data_utils import load_csv, split_train_val, validate_columns
from src.trainers import TrainConfig, train_bert, train_gnn, train_lightgbm


def parse_args():
    parser = argparse.ArgumentParser(description="Train phishing detection models (BERT / LightGBM / GNN)")

    # preferred simple workflow
    parser.add_argument("--data_path", help="Single CSV to split into train/validation")

    # optional explicit split workflow
    parser.add_argument("--train-csv", help="Training CSV path")
    parser.add_argument("--val-csv", help="Validation CSV path")

    parser.add_argument("--model", choices=["bert", "lgbm", "gnn", "all"], default="all")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--src-col", default="src_id")
    parser.add_argument("--dst-col", default="dst_id")
    parser.add_argument("--output-dir", default="runs")

    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--bert-model", default="distilbert-base-uncased")
    return parser.parse_args()


def load_dataframes(args):
    if args.data_path:
        df = load_csv(args.data_path)
        return split_train_val(df, label_col=args.label_col, val_size=args.val_size, random_state=args.random_state)

    if args.train_csv and args.val_csv:
        return load_csv(args.train_csv), load_csv(args.val_csv)

    raise ValueError("Provide either --data_path OR both --train-csv and --val-csv")


def main():
    args = parse_args()
    train_df, val_df = load_dataframes(args)

    validate_columns(train_df, [args.text_col, args.label_col])
    validate_columns(val_df, [args.text_col, args.label_col])

    cfg = TrainConfig(
        output_dir=args.output_dir,
        text_col=args.text_col,
        label_col=args.label_col,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        bert_model=args.bert_model,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    if args.model in {"bert", "all"}:
        results["bert"] = train_bert(train_df, val_df, cfg)
    if args.model in {"lgbm", "all"}:
        results["lgbm"] = train_lightgbm(train_df, val_df, cfg)
    if args.model in {"gnn", "all"}:
        validate_columns(train_df, [args.src_col, args.dst_col])
        validate_columns(val_df, [args.src_col, args.dst_col])
        results["gnn"] = train_gnn(train_df, val_df, cfg, src_col=args.src_col, dst_col=args.dst_col)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
