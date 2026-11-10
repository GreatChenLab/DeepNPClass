# src/data_utils.py
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, RDKFingerprint as RDKFP
from torch_geometric.data import Data
import numpy as np

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

def probs_to_multilabel_preds(probs, threshold=0.5):
    probs = np.asarray(probs, dtype=np.float64)
    single = probs.ndim == 1
    if single:
        probs = probs[None, :]
    preds = (probs >= threshold).astype(np.float32)
    none_pass = preds.sum(axis=1) == 0
    if np.any(none_pass):
        argmax_idx = np.argmax(probs[none_pass], axis=1)
        preds[none_pass, argmax_idx] = 1.0
    if single:
        return preds[0]
    return preds

def smiles_to_maccs(smiles_list):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append([0] * 167)
        else:
            arr = MACCSkeys.GenMACCSKeys(mol).ToList()
            fps.append(arr)
    return torch.tensor(fps, dtype=torch.float)

def smiles_to_rdkfp(smiles_list, fp_size=4096):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append([0] * fp_size)
        else:
            fps.append(list(RDKFP(mol, fpSize=fp_size)))
    return torch.tensor(fps, dtype=torch.float)

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    x = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.float).view(-1, 1)
    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])
    return Data(x=x, edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous())

class MolDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, fps, labels):
        self.graphs = graphs
        self.fps    = fps
        self.labels = labels
        assert len(self.graphs) == len(self.fps) == len(self.labels)
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.float).view(-1)
        return self.graphs[idx], self.fps[idx], label

class PredictDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, fps):
        self.graphs = graphs
        self.fps = fps
    def __len__(self): return len(self.graphs)
    def __getitem__(self, idx): return self.graphs[idx], self.fps[idx]

def prepare_training_data(train_csv_path, validation_csv_path, label_column='Pathway'):
    train_data = pd.read_csv(train_csv_path)
    validation_data = pd.read_csv(validation_csv_path)

    if label_column not in train_data.columns or label_column not in validation_data.columns:
        raise ValueError(f'The label column does not exist: {label_column!r}')
    
    train_maccs = smiles_to_maccs(train_data['index'].tolist())
    validation_maccs  = smiles_to_maccs(validation_data['index'].tolist())

    train_rdkfp = smiles_to_rdkfp(train_data['index'].tolist())
    validation_rdkfp  = smiles_to_rdkfp(validation_data['index'].tolist())

    train_fp = torch.cat([train_rdkfp, train_maccs], dim=1)
    validation_fp  = torch.cat([validation_rdkfp,  validation_maccs],  dim=1)

    train_smiles_list = train_data['index'].tolist()
    validation_smiles_list = validation_data['index'].tolist()
    train_graphs = [smiles_to_graph(s) for s in train_smiles_list]
    validation_graphs = [smiles_to_graph(s) for s in validation_smiles_list]

    valid_train_idx = [i for i, g in enumerate(train_graphs) if g is not None]
    valid_validation_idx = [i for i, g in enumerate(validation_graphs) if g is not None]

    train_graphs = [train_graphs[i] for i in valid_train_idx]
    validation_graphs = [validation_graphs[i] for i in valid_validation_idx]
    train_fp = train_fp[valid_train_idx]
    validation_fp = validation_fp[valid_validation_idx]
    train_labels = [train_data[label_column].tolist()[i] for i in valid_train_idx]
    validation_labels = [validation_data[label_column].tolist()[i] for i in valid_validation_idx]

    return train_graphs, train_fp, train_labels, validation_graphs, validation_fp, validation_labels

def prepare_prediction_data(input_csv_path, label_column=1, mlb=None):
    data_df = pd.read_csv(input_csv_path, header=None)  
    
    smiles_list = data_df.iloc[:, 0].tolist()  

    maccs = smiles_to_maccs(smiles_list)
    rdkfp = smiles_to_rdkfp(smiles_list)
    fp = torch.cat([rdkfp, maccs], dim=1)
    fp_dim = fp.shape[1]

    graphs = [smiles_to_graph(s) for s in smiles_list]
    valid_idx = [i for i, g in enumerate(graphs) if g is not None]

    graphs = [graphs[i] for i in valid_idx]
    fp = fp[valid_idx]
    valid_smiles = [smiles_list[i] for i in valid_idx]

    return valid_smiles, graphs, fp, valid_idx, fp_dim