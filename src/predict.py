# src/predict.py
import torch
import numpy as np
import pandas as pd
import pickle


def load_model(model_class, model_path, model_params, device):
    model = model_class(**model_params)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def make_predictions(model, predict_loader, class_names, device, threshold=0.5):
    all_probs = []
    all_final_classes = []
    with torch.no_grad():
        batch_count = 0
        for batch_data, fp in predict_loader:
            batch_count += 1
            print(f"The {batch_count} th batch is being processed...")

            batch_data, fp = batch_data.to(device), fp.to(device)
            out = model(batch_data, fp)
            probs = torch.sigmoid(out).cpu().numpy()
            all_probs.append(probs)

            for prob in probs:
                pred_class_indices = np.where(prob >= threshold)[0]
                if len(pred_class_indices) == 0:
                    pred_class_indices = [np.argmax(prob)]
                pred_class_names = [class_names[idx] for idx in pred_class_indices]
                all_final_classes.append(','.join(pred_class_names))
    return all_probs, all_final_classes


def load_mlb_and_class_names(mlb_path):
    with open(mlb_path, 'rb') as f:
        mlb = pickle.load(f)
    return mlb, list(mlb.classes_)


def evaluate_and_save_results(all_probs, all_final_classes, valid_smiles,
                              class_names, output_path):
    all_probs = np.vstack(all_probs)
    result_dict = {'smiles': valid_smiles, 'prediction_results': all_final_classes}
    for i, cls_name in enumerate(class_names):
        result_dict[f'{cls_name}_prob'] = all_probs[:, i]

    result_df = pd.DataFrame(result_dict)
    ordered_cols = ['smiles', 'prediction_results'] + [f'{c}_prob' for c in class_names]
    result_df = result_df[ordered_cols]
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nThe prediction results have been saved to：{output_path}")
    return result_df
