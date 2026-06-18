import sys
sys.path.append('..') 
from src import data_utils, model, train
import pickle
import torch

if __name__ == "__main__":
    DATA_DIR = "../data"
    MODEL_SAVE_DIR = "../models"
    TRAIN_CSV = f"{DATA_DIR}/train_data.csv"
    TEST_CSV = f"{DATA_DIR}/test_data.csv"
    
    train_graphs, train_fp, train_y, test_graphs, test_fp, test_y, mlb = \
        data_utils.prepare_training_data(TRAIN_CSV, TEST_CSV)
    
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
        test_graphs, test_fp, test_y,
        model_class=model.GINGGNNModel,
        model_params=model_params,
        device=device,
        save_dir=MODEL_SAVE_DIR
    )
    
    print(f"Training completed!")