# scripts/pred_main.py
import sys
sys.path.append('..')
from src import data_utils, model, predict
import torch
import pickle
import os
from torch_geometric.loader import DataLoader

if __name__ == "__main__":
    MODEL_PATH = "../models/pathway_best_model.pth" 
    MLB_PATH = "../models/mlb_pathway.pkl"
    INPUT_CSV = "../data/test_data_mini.csv"
    OUTPUT_CSV = "../results/pathway_predresults.csv"
    HAS_LABELS = False
    
    with open(MLB_PATH, 'rb') as f:
        mlb = pickle.load(f)
    class_names = list(mlb.classes_)

    valid_smiles, graphs, fp, true_labels_bin, _ = \
        data_utils.prepare_prediction_data(INPUT_CSV, mlb=mlb)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_params = {
        'node_features_dim': 1,
        'hidden_dim': 64,
        'output_dim': len(class_names),
        'fp_dim': fp.shape[1],
        'dropout_rate': 0.5
    }
    model = predict.load_model(model.GINGGNNModel, MODEL_PATH, model_params, device)
    
    predict_dataset = data_utils.PredictDataset(graphs, fp)
    predict_loader = DataLoader(predict_dataset, batch_size=64, shuffle=False)
    all_probs, all_final_classes = predict.make_predictions(model, predict_loader, class_names, device)
    
    predict.evaluate_and_save_results(
        all_probs, all_final_classes, valid_smiles,
         class_names, OUTPUT_CSV
    )
    
    print(f"Prediction completed! The result has been saved to {OUTPUT_CSV}")