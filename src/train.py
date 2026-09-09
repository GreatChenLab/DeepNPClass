# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
import torch.nn.functional as F
import scipy.stats
import os
import shutil
import random
import numpy as np
from .data_utils import MolDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    for batch_data, fp, y in train_loader:
        batch_data, fp, y = batch_data.to(device), fp.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(batch_data, fp)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)


def evaluate_on_loader(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_outputs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch_data, fp, y in data_loader:
            batch_data, fp, y = batch_data.to(device), fp.to(device), y.to(device)
            out = model(batch_data, fp)
            total_loss += criterion(out, y).item()
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).float()
            all_outputs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    all_outputs = np.array(all_outputs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cosine_sim = F.cosine_similarity(
        torch.tensor(all_outputs),
        torch.tensor(all_labels),
        dim=1
    ).mean().item()
    recalls = recall_score(all_labels, all_preds, average=None, zero_division=0)

    metrics = {
        'loss': avg_loss,
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        'prec_macro': precision_score(all_labels, all_preds, average='macro'),
        'recall_macro': recall_score(all_labels, all_preds, average='macro'),
        'recalls': recalls,
        'g_mean': scipy.stats.gmean(recalls),
        'cosine_sim': cosine_sim,
        'map': average_precision_score(all_labels, all_outputs, average='macro'),
    }
    return metrics


def train_one_fold(fold, train_loader, val_loader, model_class, model_params,
                   device, save_dir, save_prefix, max_epochs=100, patience=10):
    model = model_class(**model_params).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_epoch = -1
    best_val_loss = float('inf')
    best_val_f1 = -1.0
    best_val_prec = None
    best_val_recalls = None
    best_val_cosine_sim = None
    best_val_map = None
    counter = 0
    fold_model_path = os.path.join(save_dir, f'{save_prefix}_model_fold{fold}.pth')

    for epoch in range(max_epochs):
        avg_train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_on_loader(model, val_loader, criterion, device)
        scheduler.step(val_metrics['loss'])

        print(
            f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_metrics["loss"]:.4f}, '
            f'Macro-F1: {val_metrics["f1_macro"]:.4f}, Weighted-F1: {val_metrics["f1_weighted"]:.4f}, '
            f'Macro-Prec: {val_metrics["prec_macro"]:.4f}, G-mean: {val_metrics["g_mean"]:.4f}, '
            f'CosineSim: {val_metrics["cosine_sim"]:.4f}, mAP: {val_metrics["map"]:.4f}'
        )

        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_epoch = epoch + 1
            best_val_loss = val_metrics['loss']
            best_val_prec = val_metrics['prec_macro']
            best_val_recalls = val_metrics['recalls']
            best_val_cosine_sim = val_metrics['cosine_sim']
            best_val_map = val_metrics['map']
            torch.save(model.state_dict(), fold_model_path)
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    return {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'best_val_f1': best_val_f1,
        'best_val_prec': best_val_prec,
        'best_val_recall': best_val_recalls,
        'best_val_cosine_sim': best_val_cosine_sim,
        'best_val_map': best_val_map,
        'fold_model_path': fold_model_path,
    }


def evaluate_final_model(model_class, model_params, model_path, validation_loader, device):
    criterion = nn.BCEWithLogitsLoss()
    model = model_class(**model_params)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    metrics = evaluate_on_loader(model, validation_loader, criterion, device)

    print("\nFinal assessment result:")
    print(f"validation Macro-F1:    {metrics['f1_macro']:.4f}")
    print(f"validation Weighted-F1: {metrics['f1_weighted']:.4f}")
    print(f"validation Macro-Prec:  {metrics['prec_macro']:.4f}")
    print(f"validation Macro-Recall:  {metrics['recall_macro']:.4f}")
    print(f"validation Cosine Similarity: {metrics['cosine_sim']:.4f}")
    print(f"validation mAP:         {metrics['map']:.4f}")
    return metrics


def train_model(train_graphs, train_fp, train_y, validation_graphs, validation_fp, validation_y,
                model_class, model_params, device, save_dir, save_prefix='train',
                seed=42, **kwargs):
    set_seed(seed)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    fold_results = []
    best_val_f1_overall = -1.0
    best_fold = 1

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_graphs)):
        print(f"Fold {fold+1}")

        train_idx = [i for i in train_idx if i < len(train_fp)]
        val_idx = [i for i in val_idx if i < len(train_fp)]

        train_graphs_fold = [train_graphs[i] for i in train_idx]
        val_graphs_fold = [train_graphs[i] for i in val_idx]
        train_fp_fold = train_fp[train_idx]
        val_fp_fold = train_fp[val_idx]
        train_y_fold = train_y[train_idx]
        val_y_fold = train_y[val_idx]

        train_loader = DataLoader(
            MolDataset(train_graphs_fold, train_fp_fold, train_y_fold),
            batch_size=64, shuffle=True)
        val_loader = DataLoader(
            MolDataset(val_graphs_fold, val_fp_fold, val_y_fold),
            batch_size=64, shuffle=False)

        fold_res = train_one_fold(
            fold=fold + 1,
            train_loader=train_loader,
            val_loader=val_loader,
            model_class=model_class,
            model_params=model_params,
            device=device,
            save_dir=save_dir,
            save_prefix=save_prefix,
        )
        fold_results.append(fold_res)

        if fold_res['best_val_f1'] > best_val_f1_overall:
            best_val_f1_overall = fold_res['best_val_f1']
            best_fold = fold + 1

        print(f"Fold {fold+1} completed. Best Val F1: {fold_res['best_val_f1']:.4f} "
              f"(epoch {fold_res['best_epoch']})\n")

    print("\n" + "=" * 50)
    print("Five-fold Cross-Validation Results Summary:")
    for fold, res in enumerate(fold_results):
        print(f"Fold {fold+1} - Best epoch: {res['best_epoch']}, "
              f"Best Val F1: {res['best_val_f1']:.4f}, "
              f"Val Loss@best: {res['best_val_loss']:.4f}, "
              f"mAP@best: {res['best_val_map']:.4f}, "
              f"CosineSim@best: {res['best_val_cosine_sim']:.4f}")
    print(f"\nThe best-performing model comes from Fold {best_fold} "
          f"(Macro-F1: {best_val_f1_overall:.4f})")
    print("=" * 50)

    best_fold_path = os.path.join(save_dir, f'{save_prefix}_model_fold{best_fold}.pth')
    final_model_path = os.path.join(save_dir, f'{save_prefix}_best_model.pth')
    shutil.copy2(best_fold_path, final_model_path)
    print(f"\nLoad the model with the best performance: {best_fold_path}")
    print(f"Copied to: {final_model_path}")

    validation_loader = DataLoader(
        MolDataset(validation_graphs, validation_fp, validation_y),
        batch_size=64, shuffle=False)
    print(f"Best Model from Fold: {best_fold}")
    evaluate_final_model(model_class, model_params, best_fold_path, validation_loader, device)

    return best_fold
