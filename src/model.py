# src/model.py
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, GatedGraphConv, global_add_pool, MLP

class GINGGNNModel(nn.Module):
    def __init__(self, node_features_dim, hidden_dim, output_dim,
                 fp_dim, dropout_rate=0.5):
        super(GINGGNNModel, self).__init__()
        self.gin_mlp = MLP([node_features_dim, hidden_dim*2, hidden_dim*2],
                           dropout=dropout_rate, act='relu')
        self.gin = GINConv(self.gin_mlp)
        self.gin_bn = nn.BatchNorm1d(hidden_dim*2)

        self.ggnn = GatedGraphConv(hidden_dim*2, num_layers=7)
        self.ggnn_bn = nn.BatchNorm1d(hidden_dim*2)

        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim*2 + fp_dim, hidden_dim*4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim*4),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim*2),
            nn.Dropout(dropout_rate)
        )
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, data, fp):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gin(x, edge_index)
        x = self.gin_bn(x)
        x = self.ggnn(x, edge_index)
        x = self.ggnn_bn(x)
        global_pooling = global_add_pool(x, batch)
        combined = torch.cat((global_pooling, fp), dim=-1)
        features = self.shared_fc(combined)
        return self.fc(features).squeeze()

