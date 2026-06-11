# DeepNPClass: A Hybrid Deep Learning Model Integrating Molecular Graph and Fingerprint Features for Hierarchical Natural Product Classification
DeepNPClass is a deep learning-based tool that automatically classify natural product molecules. Given a CSV file containing SMILES strings, this tool predicts their chemical classes (e.g., alkaloids, flavonoids, terpenoids, etc.).

# Environment
1.The project relies on dependencies specified in requirements.txt. We recommend using Conda to manage the environment.
Create and activate a new Conda environment (Python 3.8 is recommended):
```
conda create -n deepnpclass python=3.8 -y
conda activate deepnpclass
```
2.Install all dependencies from the requirements.txt file:
pip install -r requirements.txt

# Prepare Your Input Data
Create a CSV file (e.g., test_data_mini.csv) that contains a column named smiles with your SMILES strings.
```
SMILES
C=C1CC23CC1CCC2C1(C)CCCC(C)(C(=O)O)C1C3C(=O)O
CCCCCCCC/C=C\CCCCCCCCCC(=O)O[C@H](COC(=O)CCCCCCC/C=C\CCCCCCCCC)COP(=O)(O)OC[C@@H](O)CO
...
```

# Run the Prediction
Place your input file in the data/ directory and execute the prediction script.
```
python -m scripts.pred_main
```

The prediction results will be saved to results/predictions.csv.

# Configuration Options
You can customize the behavior by editing the configuration section at the top of the scripts/pred_main.py file:
```
INPUT_CSV = "data/input.csv"    # Path to the input file
MODEL_PATH = "models/best_model.pth" # Path to the pre-trained model
MLB_PATH = "models/mlb.pkl"     # Path to the label encoder
OUTPUT_CSV = "results/predictions.csv" # Path to the output file
```
Simply modify the corresponding labels to switch between different levels of predictions.
