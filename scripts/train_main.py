# scripts/train_main.py
# 作用：按层级训练（pathway / superclass / class）；输出带 train_ 前缀，不覆盖已有预训练文件
# 用法：python scripts/train_main.py --level pathway
import os
import sys
import argparse
import pickle

import torch
from sklearn.preprocessing import MultiLabelBinarizer

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import data_utils, model, train
from src.paths import get_data_dir, get_models_dir

# 三个层级的标签列与输出命名
LEVEL_CONFIG = {
    'pathway': {
        'label_column': 'Pathway',
        'save_prefix': 'train_pathway',
        'mlb_name': 'train_mlb_pathway.pkl',
    },
    'superclass': {
        'label_column': 'Super_class',
        'save_prefix': 'train_superclass',
        'mlb_name': 'train_mlb_superclass.pkl',
    },
    'class': {
        'label_column': 'Class',
        'save_prefix': 'train_class',
        'mlb_name': 'train_mlb_class.pkl',
    },
}


def run_training(level, seed=27):
    """按指定层级跑完整训练流程。"""
    if level not in LEVEL_CONFIG:
        raise ValueError(f'未知层级: {level!r}，可选 {list(LEVEL_CONFIG)}')

    cfg = LEVEL_CONFIG[level]
    label_column = cfg['label_column']
    save_prefix = cfg['save_prefix']
    mlb_name = cfg['mlb_name']

    # 各层级训练前都固定随机种子
    train.set_seed(seed)

    data_dir = get_data_dir()
    model_save_dir = get_models_dir()
    train_csv = os.path.join(data_dir, 'train_data.csv')
    validation_csv = os.path.join(data_dir, 'validation_data.csv')

    train_graphs, train_fp, train_labels, validation_graphs, validation_fp, validation_labels = \
        data_utils.prepare_training_data(train_csv, validation_csv, label_column=label_column)

    mlb = MultiLabelBinarizer()
    train_y = mlb.fit_transform([[l] for l in train_labels])
    validation_y = mlb.transform([[l] for l in validation_labels])

    # 新建 train_mlb_*，不覆盖已有 mlb_pathway.pkl 等
    mlb_path = os.path.join(model_save_dir, mlb_name)
    with open(mlb_path, 'wb') as f:
        pickle.dump(mlb, f)
    print(f'Saved mlb to: {mlb_path}')

    model_params = {
        'node_features_dim': 1,
        'hidden_dim': 64,
        'output_dim': len(mlb.classes_),
        'fp_dim': train_fp.shape[1],
        'dropout_rate': 0.5,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_fold = train.train_model(
        train_graphs, train_fp, train_y,
        validation_graphs, validation_fp, validation_y,
        model_class=model.GINGGNNModel,
        model_params=model_params,
        device=device,
        save_dir=model_save_dir,
        save_prefix=save_prefix,
        seed=seed,
    )

    print(f'{level} training completed! Best fold: {best_fold}')
    print(f'Models: {save_prefix}_model_fold*.pth / {save_prefix}_best_model.pth')
    return best_fold


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DeepNPClass by hierarchy level')
    parser.add_argument(
        '--level',
        type=str,
        required=True,
        choices=['pathway', 'superclass', 'class'],
        help='训练层级',
    )
    parser.add_argument('--seed', type=int, default=27, help='随机种子')
    args = parser.parse_args()
    run_training(args.level, seed=args.seed)
