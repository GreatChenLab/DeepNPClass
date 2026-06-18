# scripts/pred_main.py
import sys
sys.path.append('..')
from src import data_utils, model, predict
import torch
import pickle
import os
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader

def run_prediction_for_level(model_path, mlb_path, graphs, fp, device, model_class):
    mlb, class_names = predict.load_mlb_and_class_names(mlb_path)
    model_params = {
        'node_features_dim': 1,
        'hidden_dim': 64,
        'output_dim': len(class_names),
        'fp_dim': fp.shape[1], 
        'dropout_rate': 0.5
    }
    
    loaded_model = predict.load_model(model_class, model_path, model_params, device)
    
    predict_dataset = data_utils.PredictDataset(graphs, fp)
    predict_loader = DataLoader(predict_dataset, batch_size=64, shuffle=False)
    
    all_probs, all_final_classes = predict.make_predictions(loaded_model, predict_loader, class_names, device)
    
    all_probs = np.vstack(all_probs)
    
    return all_probs, all_final_classes, class_names

if __name__ == "__main__":
    INPUT_CSV = "../data/example.csv"
    OUTPUT_CSV = "../results/example_predictions.csv"
    LEVELS = ['pathway', 'superclass', 'class']
    MODEL_CONFIG = {
                'pathway': {
            'model_path': '../models/pathway_best_model.pth',
            'mlb_path': '../models/mlb_pathway.pkl'
        },
        'superclass': {
            'model_path': '../models/superclass_best_model.pth',
            'mlb_path': '../models/mlb_superclass.pkl'
        },
        'class': {
            'model_path': '../models/class_best_model.pth',
            'mlb_path': '../models/mlb_class.pkl'
        }
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(MODEL_CONFIG['pathway']['mlb_path'], 'rb') as f:
        temp_mlb = pickle.load(f)
    valid_smiles, graphs, fp, _, _ = data_utils.prepare_prediction_data(INPUT_CSV)

    print(f"Start to predict for {len(valid_smiles)} valid SMILES...")
    
    result_dict = {'SMILES': valid_smiles}
    
    prob_columns_order = []
    
    for level in LEVELS:
        print(f"Currently predicting at the {level}...")
        probs, predictions, class_names = run_prediction_for_level(
            MODEL_CONFIG[level]['model_path'],
            MODEL_CONFIG[level]['mlb_path'],
            graphs, fp, device,
            model.GINGGNNModel  
        )
        
        result_dict[f'{level}_prediction_results'] = predictions
        
        for i, cls_name in enumerate(class_names):
            col_name = f'{level}_{cls_name}_prob'
            result_dict[col_name] = probs[:, i]
            prob_columns_order.append(col_name)
    
    final_columns = ['SMILES']
    final_columns.extend([f'{level}_prediction_results' for level in LEVELS])
    final_columns.extend(prob_columns_order)
    
    result_df = pd.DataFrame(result_dict)
    result_df = result_df[final_columns]
    
    result_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    print(f"Prediction completed! The result has been saved to {OUTPUT_CSV}")

