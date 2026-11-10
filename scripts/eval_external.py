import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import data_utils, model, predict
from src.data_utils import probs_to_multilabel_preds
from src.paths import get_data_dir, get_models_dir, get_results_dir

LEVELS = {
    'pathway': {
        'csv': 'pathway_subset.csv',
        'label_col': 'pathway',
        'model': 'pathway_model.pth',
        'mlb': 'mlb_pathway.pkl',
    },
    'superclass': {
        'csv': 'superclass_subset.csv',
        'label_col': 'superclass',
        'model': 'superclass_model.pth',
        'mlb': 'mlb_superclass.pkl',
    },
    'class': {
        'csv': 'class_subset.csv',
        'label_col': 'class',
        'model': 'class_model.pth',
        'mlb': 'mlb_class.pkl',
    },
}


def prepare_subset(csv_path, label_col):
    df = pd.read_csv(csv_path)
    smiles = df['smiles'].tolist()
    labels = df[label_col].astype(str).tolist()
    maccs = data_utils.smiles_to_maccs(smiles)
    rdkfp = data_utils.smiles_to_rdkfp(smiles)
    fp = torch.cat([rdkfp, maccs], dim=1)
    graphs = [data_utils.smiles_to_graph(s) for s in smiles]
    keep = [i for i, g in enumerate(graphs) if g is not None]
    graphs = [graphs[i] for i in keep]
    fp = fp[keep]
    smiles = [smiles[i] for i in keep]
    labels = [labels[i] for i in keep]
    return smiles, graphs, fp, labels


def predict_level(graphs, fp, model_path, mlb_path, device):
    _, class_names = predict.load_mlb_and_class_names(mlb_path)
    net = predict.load_model(
        model.GINGGNNModel,
        model_path,
        {
            'node_features_dim': 1,
            'hidden_dim': 64,
            'output_dim': len(class_names),
            'fp_dim': fp.shape[1],
            'dropout_rate': 0.5,
        },
        device,
    )
    loader = DataLoader(data_utils.PredictDataset(graphs, fp), batch_size=64, shuffle=False)
    probs_list = []
    with torch.no_grad():
        for batch_data, batch_fp in loader:
            batch_data = batch_data.to(device)
            batch_fp = batch_fp.to(device)
            out = net(batch_data, batch_fp)
            probs = torch.sigmoid(out).cpu().numpy()
            if probs.ndim == 1:
                probs = probs[None, :]
            probs_list.append(probs)
    probs = np.vstack(probs_list)
    preds = probs_to_multilabel_preds(probs, threshold=0.5)
    pred_names = []
    for row in preds:
        names = [class_names[i] for i, v in enumerate(row) if v > 0]
        pred_names.append(','.join(names))
    return pred_names


def run_level(level, dedup_dir, model_dir, device):
    cfg = LEVELS[level]
    smiles, graphs, fp, labels = prepare_subset(
        os.path.join(dedup_dir, cfg['csv']), cfg['label_col'])
    pred_names = predict_level(
        graphs, fp,
        os.path.join(model_dir, cfg['model']),
        os.path.join(model_dir, cfg['mlb']),
        device,
    )
    pred_df = pd.DataFrame({
        'smiles': smiles,
        'ground_truth': labels,
        'prediction_results': pred_names,
    })
    print(f'[{level}] n={len(pred_df)}')
    return pred_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--level', type=str, default='all',
        choices=['all', 'pathway', 'superclass', 'class'])
    parser.add_argument(
        '--data_dir', type=str,
        default=os.path.join(
            get_data_dir(), 'independent external test set', 'inchikey_deduplicated'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir = get_models_dir()
    out_dir = get_results_dir()
    levels = ['pathway', 'superclass', 'class'] if args.level == 'all' else [args.level]

    for level in levels:
        pred_df = run_level(level, args.data_dir, model_dir, device)
        pred_path = os.path.join(out_dir, f'external_{level}_predictions.csv')
        pred_df.to_csv(pred_path, index=False, encoding='utf-8-sig')
        print(f'saved={pred_path}')


if __name__ == '__main__':
    main()
