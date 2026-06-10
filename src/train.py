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

def train_model(train_graphs, train_fp, train_y, test_graphs, test_fp, test_y, 
                model_class, model_params, device, save_dir, **kwargs):
    """
    执行五折交叉验证训练。
    Args:
        ... (各种输入)
        save_dir: 保存最佳模型的目录
    Returns:
        best_fold: 表现最好的折数
    """
    # 注意：将硬编码的路径 '/home/maoyufei/37_fold{fold+1}.pth' 替换为 os.path.join(save_dir, f'model_fold{fold+1}.pth')
    kf = KFold(n_splits=5, shuffle=True, random_state=27)
    fold_results = []
    best_val_f1_overall = -1  # 用于记录全局最高的 F1
    best_fold = 1           # 用于记录 F1 最高的那一折

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_graphs)):
        print(f"Fold {fold+1}")

        train_idx = [i for i in train_idx if i < len(train_fp)]
        val_idx   = [i for i in val_idx   if i < len(train_fp)]

        train_graphs_fold = [train_graphs[i] for i in train_idx]
        val_graphs_fold   = [train_graphs[i] for i in val_idx]
        train_fp_fold     = train_fp[train_idx]
        val_fp_fold       = train_fp[val_idx]
        train_y_fold      = train_y[train_idx]
        val_y_fold        = train_y[val_idx]

        train_dataset = MolDataset(train_graphs_fold, train_fp_fold, train_y_fold)
        val_dataset   = MolDataset(val_graphs_fold,   val_fp_fold,   val_y_fold)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)

        node_features_dim = 1
        hidden_dim = 64
        output_dim = len(mlb.classes_)
        fp_dim     = train_fp.shape[1]
        dropout_rate = 0.5

        # model = model_class(node_features_dim, hidden_dim, output_dim,
        #                  fp_dim, dropout_rate)
        model = model_class(**model_params)

        criterion  = nn.BCEWithLogitsLoss()
        optimizer  = optim.Adam(model.parameters(), lr=0.001)
        scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        best_val_loss = float('inf')
        best_val_f1_for_this_fold = 0.0  # 记录当前折的最佳 F1
        patience = 10
        counter  = 0

        for epoch in range(100):
            model.train()
            train_loss = 0
            for batch_data, fp, y in train_loader:
                batch_data, fp, y = batch_data.to(device), fp.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(batch_data, fp)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)

            model.eval()
            val_loss = 0
            all_val_outputs, all_val_preds, all_val_labels = [], [], []
            with torch.no_grad():
                for batch_data, fp, y in val_loader:
                    batch_data, fp, y = batch_data.to(device), fp.to(device), y.to(device)
                    out = model(batch_data, fp)
                    val_loss += criterion(out, y).item()
                    probs = torch.sigmoid(out)
                    preds = (probs > 0.5).float()
                    all_val_outputs.extend(probs.cpu().numpy())
                    all_val_preds.extend(preds.cpu().numpy())
                    all_val_labels.extend(y.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            all_val_outputs = np.array(all_val_outputs)
            all_val_preds  = np.array(all_val_preds)
            all_val_labels = np.array(all_val_labels)

            val_cosine_sim = F.cosine_similarity(
            torch.tensor(all_val_outputs), 
                torch.tensor(all_val_labels), 
                dim=1
            ).mean().item()

            val_map = average_precision_score(all_val_labels, all_val_outputs, average='macro')

            val_f1_macro  = f1_score(all_val_labels, all_val_preds, average='macro')
            val_f1_weight = f1_score(all_val_labels, all_val_preds, average='weighted')
            val_prec      = precision_score(all_val_labels, all_val_preds, average='macro')
            val_recalls   = recall_score(all_val_labels, all_val_preds, average=None, zero_division=0)
            g_mean        = scipy.stats.gmean(val_recalls)

            print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
              f'Macro-F1: {val_f1_macro:.4f}, Weighted-F1: {val_f1_weight:.4f}, '
              f'Macro-Prec: {val_prec:.4f}, G-mean: {g_mean:.4f}, '
              f'CosineSim: {val_cosine_sim:.4f}, mAP: {val_map:.4f}')

            # --- 同时监控 Val Loss 和 Val F1 ---
            # 如果当前验证F1比这折之前记录的最好F1还要高，则更新并保存模型
            if val_f1_macro > best_val_f1_for_this_fold:
                best_val_f1_for_this_fold = val_f1_macro
                # 保存模型
                model_path = os.path.join(save_dir, f'model_fold{fold+1}.pth')
                torch.save(model.state_dict(), model_path)
                counter = 0
            else:
                counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        # 记录该折的最终最佳结果
        fold_results.append({
        'best_epoch': epoch,
        'best_val_loss': best_val_loss,
        'best_val_f1': best_val_f1_for_this_fold,  # 使用这折的最佳F1
        'best_val_prec': val_prec,
        'best_val_recall': val_recalls,
        'best_val_cosine_sim': val_cosine_sim,
        'best_val_map': val_map
        })

    # --- 更新全局最佳模型信息 ---
        if best_val_f1_for_this_fold > best_val_f1_overall:
            best_val_f1_overall = best_val_f1_for_this_fold
            best_fold = fold + 1  # fold是0索引，所以+1

        print(f"Fold {fold+1} completed. Best Val F1: {best_val_f1_for_this_fold:.4f}\n")

    # 打印五折结果
    print("\n" + "="*50)
    print("五折交叉验证结果汇总:")
    for fold, res in enumerate(fold_results):
        print(f"Fold {fold+1} - Best Val F1: {res['best_val_f1']:.4f}, "
          f"Best Val mAP: {res['best_val_map']:.4f}, "
          f"Best Val CosineSim: {res['best_val_cosine_sim']:.4f}")
    print(f"\n表现最好的模型来自 Fold {best_fold} (Macro-F1: {best_val_f1_overall:.4f})")
    print("="*50)

    # -------------------- 最终测试评估 --------------------
    test_dataset = MolDataset(test_graphs, test_fp, test_y)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # --- 加载 F1 最高的那一折的模型 ---
    best_model_path = os.path.join(save_dir, f'model_fold{best_fold}.pth')
    print(f"\n加载表现最好的模型: {best_model_path}")

    # 重新实例化模型（或确保模型结构一致）
    # model = GINGGNNModel(node_features_dim, hidden_dim, output_dim,
    #                  fp_dim, dropout_rate)
    model = model_class(**model_params)  
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    all_test_outputs, all_test_preds, all_test_labels = [], [], []
    with torch.no_grad():
        for batch_data, fp, y in test_loader:
            batch_data, fp, y = batch_data.to(device), fp.to(device), y.to(device)
            out = model(batch_data, fp)
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).float()
            all_test_outputs.extend(probs.cpu().numpy())
            all_test_preds.extend(preds.cpu().numpy())
            all_test_labels.extend(y.cpu().numpy())

    all_test_outputs = np.array(all_test_outputs)
    all_test_preds  = np.array(all_test_preds)
    all_test_labels = np.array(all_test_labels)

    test_cosine_sim = F.cosine_similarity(
        torch.tensor(all_test_outputs), 
        torch.tensor(all_test_labels), 
        dim=1
    ).mean().item()

    test_map = average_precision_score(all_test_labels, all_test_outputs, average='macro')

    test_f1_macro  = f1_score(all_test_labels, all_test_preds, average='macro')
    test_f1_weight = f1_score(all_test_labels, all_test_preds, average='weighted')
    test_prec      = precision_score(all_test_labels, all_test_preds, average='macro')
    test_recall_macro = recall_score(all_test_labels, all_test_preds, average='macro')

    print("\n最终评估结果:")
    print(f"Best Model from Fold: {best_fold}")
    print(f"Test Macro-F1:    {test_f1_macro:.4f}")
    print(f"Test Weighted-F1: {test_f1_weight:.4f}")
    print(f"Test Macro-Prec:  {test_prec:.4f}")
    print(f"Test Recall (macro):  {test_recall_macro:.4f}")
    print(f"Test Cosine Similarity: {test_cosine_sim:.4f}")
    print(f"Test mAP:         {test_map:.4f}")

    return best_fold