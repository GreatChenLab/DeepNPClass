# src/data_utils.py
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, RDKFingerprint as ExtFP
from torch_geometric.data import Data
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# 忽略RDKit警告
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# --- 特征处理函数 ---
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

def smiles_to_extfp(smiles_list, fp_size=4096):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append([0] * fp_size)
        else:
            fps.append(list(ExtFP(mol, fpSize=fp_size)))
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

# --- Dataset类 ---
class MolDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, fps, labels):
        self.graphs = graphs
        self.fps    = fps
        self.labels = labels
        assert len(self.graphs) == len(self.fps) == len(self.labels), "数据长度不一致"
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        return self.graphs[idx], self.fps[idx], torch.tensor(self.labels[idx], dtype=torch.float)

class PredictDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, fps):
        self.graphs = graphs
        self.fps = fps
    def __len__(self): return len(self.graphs)
    def __getitem__(self, idx): return self.graphs[idx], self.fps[idx]

# --- 高层数据准备函数 ---
def prepare_training_data(train_csv_path, test_csv_path):
    """
    加载CSV，生成图、指纹、标签，并返回处理好的训练/测试数据及mlb。
    """
    train_data = pd.read_csv(train_csv_path)
    test_data = pd.read_csv(test_csv_path)
    
    train_maccs = smiles_to_maccs(train_data['index'].tolist())
    test_maccs  = smiles_to_maccs(test_data['index'].tolist())

    train_extfp = smiles_to_extfp(train_data['index'].tolist())
    test_extfp  = smiles_to_extfp(test_data['index'].tolist())

    train_fp = torch.cat([train_extfp, train_maccs], dim=1)
    test_fp  = torch.cat([test_extfp,  test_maccs],  dim=1)

    train_smiles_list = train_data['index'].tolist()
    train_graphs = [smiles_to_graph(s) for s in train_smiles_list if smiles_to_graph(s)]
    test_graphs = [smiles_to_graph(s) for s in test_data['index'].tolist() if smiles_to_graph(s)]

    # 对齐索引
    valid_train_idx = [i for i, g in enumerate(train_graphs) if g is not None]
    valid_test_idx  = [i for i, g in enumerate(test_graphs)  if g is not None]

    valid_train_idx = [idx for idx in valid_train_idx if idx < len(train_fp)]
    valid_test_idx  = [idx for idx in valid_test_idx  if idx < len(test_fp)]

    train_fp   = train_fp[valid_train_idx]
    test_fp    = test_fp[valid_test_idx]
    train_labels = [train_data['Pathway'].tolist()[i] for i in valid_train_idx]
    test_labels  = [test_data['Pathway'].tolist()[i]  for i in valid_test_idx]

    # 标签二值化
    mlb = MultiLabelBinarizer()
    train_y = mlb.fit_transform([[l] for l in train_labels])
    test_y  = mlb.transform([[l] for l in test_labels])

    return train_graphs, train_fp, train_y, test_graphs, test_fp, test_y, mlb

def prepare_prediction_data(input_csv_path, label_column=1, mlb=None):
    """
    为预测准备数据。
    """
    data_df = pd.read_csv(input_csv_path) 
    smiles_list = data_df['SMILES'].tolist()   # 假设SMILES列名为'SMILES'

    maccs = smiles_to_maccs(smiles_list)
    extfp = smiles_to_extfp(smiles_list)
    fp = torch.cat([extfp, maccs], dim=1)
    fp_dim = fp.shape[1]

   #检查有效数据
    graphs = [smiles_to_graph(s) for s in smiles_list]
    valid_idx = [i for i, g in enumerate(graphs) if g is not None]
    invalid_idx = [i for i, g in enumerate(graphs) if g is None]  # 新增：无效数据索引
    print(f"原始数据 {len(smiles_list)} 个，预处理后有效数据 {len(valid_idx)} 个，无效数据 {len(invalid_idx)} 个。")

    graphs = [graphs[i] for i in valid_idx]
    fp = fp[valid_idx]
    valid_smiles = [smiles_list[i] for i in valid_idx]

    return valid_smiles, graphs, fp, valid_idx, fp_dim