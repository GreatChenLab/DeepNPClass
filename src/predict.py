# src/predict.py
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, average_precision_score
import torch.nn.functional as F
import pickle
import os

def load_model(model_class, model_path, model_params, device):
    """加载训练好的模型"""
    model = model_class(**model_params)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def make_predictions(model, predict_loader, class_names, device, threshold=0.5):
    """执行预测并返回概率和最终类别"""
    print(f"预测DataLoader中包含 {len(predict_loader)} 个批次。")

    all_probs = []
    all_final_classes = []

    with torch.no_grad():
        batch_count = 0
        for batch_data, fp in predict_loader:
            batch_count += 1
            print(f"正在处理第 {batch_count} 个批次...") # 【调试输出4】
        
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

    if not all_probs:
        print("错误：预测循环未生成任何概率数据。")
    return all_probs, all_final_classes

def evaluate_and_save_results(all_probs, all_final_classes, valid_smiles, 
                              class_names, output_path):
    """保存结果到CSV"""
    all_probs = np.vstack(all_probs)
    # -------------------- 保存最终结果 --------------------
    result_dict = {}
    for i, cls_name in enumerate(class_names):
        result_dict[f'{cls_name}_prob'] = all_probs[:, i]
    result_dict['prediction_results'] = all_final_classes
    result_dict['smiles'] = valid_smiles

    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n预测结果已保存至：{output_path}")
    pass