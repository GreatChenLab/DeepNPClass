# scripts/pred_main.py
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import data_utils, model, predict
from src.paths import get_data_dir, get_models_dir, get_results_dir
import torch
import pickle
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
    DATA_DIR = get_data_dir()
    MODEL_DIR = get_models_dir()
    RESULTS_DIR = get_results_dir()
    INPUT_CSV = os.path.join(DATA_DIR, "example.csv")
    OUTPUT_CSV = os.path.join(RESULTS_DIR, "example_predictions.csv")
    LEVELS = ['pathway', 'superclass', 'class']
    MODEL_CONFIG = {
        'pathway': {
            'model_path': os.path.join(MODEL_DIR, 'pathway_model.pth'),
            'mlb_path': os.path.join(MODEL_DIR, 'mlb_pathway.pkl')
        },
        'superclass': {
            'model_path': os.path.join(MODEL_DIR, 'superclass_model.pth'),
            'mlb_path': os.path.join(MODEL_DIR, 'mlb_superclass.pkl')
        },
        'class': {
            'model_path': os.path.join(MODEL_DIR, 'class_model.pth'),
            'mlb_path': os.path.join(MODEL_DIR, 'mlb_class.pkl')
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

