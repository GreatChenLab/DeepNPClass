# DeepNPClass

A hybrid deep learning model integrating molecular graph and fingerprint features for natural product classification (Pathway / Superclass / Class).

Given a CSV of SMILES strings, DeepNPClass predicts chemical categories such as alkaloids, flavonoids, and terpenoids.

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
│       └── class_subset.csv        
├── models/                         
├── results/                        # prediction outputs
├── scripts/
│   ├── pred_main.py             # prediction entry
│   └── train_main.py            # training entry (--level)
└── src/                         
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

From the **project root**:

```bash
python scripts/pred_main.py
```

### Output

Results are written to:

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

## License

See `LICENSE`.
