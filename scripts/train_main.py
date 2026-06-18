import sys
sys.path.append('..') 
from src import data_utils, model, train
import pickle
import torch

if __name__ == "__main__":
    DATA_DIR = "../data"
    MODEL_SAVE_DIR = "../models"
    TRAIN_CSV = f"{DATA_DIR}/train_data.csv"
    validation_CSV = f"{DATA_DIR}/validation_data.csv"
    
    train_graphs, train_fp, train_y, validation_graphs, validation_fp, validation_y, mlb = \
        data_utils.prepare_training_data(TRAIN_CSV, validation_CSV)
    
    with open(f"{MODEL_SAVE_DIR}/mlb.pkl", 'wb') as f:
        pickle.dump(mlb, f)
    
    model_params = {
        'node_features_dim': 1,
        'hidden_dim': 64,
        'output_dim': len(mlb.classes_),
        'fp_dim': train_fp.shape[1],
        'dropout_rate': 0.5
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_fold = train.train_model(
        train_graphs, train_fp, train_y,
        validation_graphs, validation_fp, validation_y,
        model_class=model.GINGGNNModel,
        model_params=model_params,
        device=device,
        save_dir=MODEL_SAVE_DIR
    )
    
    print(f"Training completed!")