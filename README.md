# DeepNPClass

A hybrid deep learning model integrating molecular graph and fingerprint features for natural product classification (Pathway / Superclass / Class).

## Environment

Python 3.8+ is recommended. Create and activate a Conda environment:

```bash
conda create -n deepnpclass python=3.8 -y
conda activate deepnpclass
pip install -r requirements.txt
```

## Project layout

```
DeepNPClass-main/
├── data/
│   ├── example.csv                 # prediction input example
│   ├── train_data.csv
│   ├── validation_data.csv
│   ├── NPClassifier_dataset.xlsx
│   └── independent external test set/
│       ├── pathway_subset.csv
│       ├── superclass_subset.csv
│       ├── class_subset.csv
│       └── inchikey_deduplicated/  # InChIKey-deduplicated subsets
│           ├── pathway_subset.csv
│           ├── superclass_subset.csv
│           ├── class_subset.csv
│           └── dedup_by_inchikey.py
├── models/
│   ├── pathway_model.pth
│   ├── superclass_model.pth
│   ├── class_model.pth
│   ├── mlb_pathway.pkl
│   ├── mlb_superclass.pkl
│   └── mlb_class.pkl
├── results/                        # prediction / evaluation outputs
├── scripts/
│   ├── pred_main.py             # prediction entry
│   ├── train_main.py            # training entry (--level)
│   └── eval_external.py         # external test evaluation
└── src/
    └── attention_model.py       # attention-augmented architecture (interpretability)
```

## Data preprocessing

Released `train_data.csv` / `validation_data.csv` are already split (62,411 / 15,925).  
During training and inference, each SMILES is converted to:

- an undirected molecular graph with atomic numbers as node features;
- a concatenated fingerprint vector: RDKit topological fingerprint (4,096 bits) + MACCS (167 bits).

Label columns used by level: `Pathway`, `Super_class`, `Class`.  
Invalid SMILES that cannot be parsed into graphs are dropped before training or prediction.

## Reproduce principal results

From the **project root**, after installing dependencies:

```bash
# 1) Run inference with released checkpoints
python scripts/pred_main.py
# -> results/example_predictions.csv

# 2) Predict on InChIKey-deduplicated external subsets
python scripts/eval_external.py
# -> results/external_{pathway,superclass,class}_predictions.csv

# 3) (Optional) Retrain from released data
python scripts/train_main.py --level pathway
python scripts/train_main.py --level superclass
python scripts/train_main.py --level class
# -> models/{pathway,superclass,class}_model.pth and mlb_*.pkl
```

Expected released checkpoints under `models/`:

```text
pathway_model.pth / superclass_model.pth / class_model.pth
mlb_pathway.pkl / mlb_superclass.pkl / mlb_class.pkl
```

## Prediction

### Input

Put a CSV under `data/` (e.g. `data/example.csv`). The first column should be SMILES:

```text
C=C1CC23CC1CCC2C1(C)CCCC(C)(C(=O)O)C1C3C(=O)O
CCCCCCCC/C=C\CCCCCCCCCC(=O)O[C@H](COC(=O)CCCCCCC/C=C\CCCCCCCCC)COP(=O)(O)OC[C@@H](O)CO
```

Default input path in the script: `data/example.csv`.

### Run

```bash
python scripts/pred_main.py
```

### Output

```text
results/example_predictions.csv
```

The file includes SMILES, predicted labels for pathway / superclass / class, and per-class probabilities.

## Training

Training is shared through one entry script. Choose the hierarchy level with `--level`.

```bash
python scripts/train_main.py --level pathway
python scripts/train_main.py --level superclass
python scripts/train_main.py --level class
```

Training and inference use the same naming convention: `{level}_model.pth` with `mlb_{level}.pkl`.

## Independent external test set (InChIKey deduplication)

Raw external subsets are under `data/independent external test set/`.  
For evaluation, molecules matching the training or validation set by **full InChIKey** were removed. Deduplicated files (pathway 1394 + superclass 1380 + class 5073 = **7847**) are in:

```text
data/independent external test set/inchikey_deduplicated/
```

Rebuild them from the project root with:

```bash
python "data/independent external test set/inchikey_deduplicated/dedup_by_inchikey.py"
```

### External evaluation

Run prediction on the InChIKey-deduplicated subsets:

```bash
python scripts/eval_external.py
python scripts/eval_external.py --level pathway
```

Outputs:

```text
results/external_pathway_predictions.csv
results/external_superclass_predictions.csv
results/external_class_predictions.csv
```

## Attention interpretability

Atom-level attention analysis in the paper uses a **separately trained** attention-augmented model, distinct from the released Pathway/Superclass/Class predictors.  
This repository releases the architecture in `src/attention_model.py` for reference. Reproducing the main classification results does **not** require the attention model.

## License

See `LICENSE`.