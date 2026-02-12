# Phishing Detection: BERT + LightGBM + GNN

Train phishing detection models from your CSV dataset using:
- **BERT** for text semantics
- **LightGBM** for text-derived/tabular style features
- **GNN** for relationship/graph signals

## 1) Install

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

If Torch/CUDA is too heavy, install CPU-only torch first and then install the remaining dependencies.

## 2) Dataset format

Required columns:
- `text`
- `label` (`0` legitimate, `1` phishing)

For GNN (or `--model all`) also include:
- `src_id`
- `dst_id`

Example:

```csv
text,label,src_id,dst_id
"verify your account now",1,a.com,b.com
"monthly report attached",0,corp.local,mail.local
```

## 3) Run training

### Easiest way (single CSV)

```bash
python train.py --data_path data/my_dataset.csv
```

By default this runs **all 3 models** and splits your dataset into train/validation.

### Run a specific model

```bash
python train.py --data_path data/my_dataset.csv --model bert
python train.py --data_path data/my_dataset.csv --model lgbm
python train.py --data_path data/my_dataset.csv --model gnn
```

### Use explicit train/validation files

```bash
python train.py --train-csv data/train.csv --val-csv data/val.csv --model all
```

## 4) Outputs

Artifacts are written under `runs/` by default:
- `runs/bert/model_bert.pth`
- `runs/lgbm/lightgbm_model.txt`
- `runs/gnn/gnn_model.pt`
- Per-model `metrics.json`

Metrics include: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`.
