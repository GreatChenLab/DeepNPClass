import torch
import torch.nn as nn
from torch.nn import LayerNorm, MultiheadAttention
from torch_geometric.nn import GINConv, GatedGraphConv, MLP, global_add_pool


class GINGGNNAttnModel(nn.Module):
    def __init__(self, node_features_dim, hidden_dim, output_dim, fp_dim, dropout_rate=0.5):
        super().__init__()
        self.attn_dim = hidden_dim * 2
        self.gin_mlp = MLP([node_features_dim, hidden_dim * 2, hidden_dim * 2],
                           dropout=dropout_rate, act='relu')
        self.gin = GINConv(self.gin_mlp)
        self.gin_bn = nn.BatchNorm1d(hidden_dim * 2)
        self.ggnn = GatedGraphConv(hidden_dim * 2, num_layers=7)
        self.ggnn_bn = nn.BatchNorm1d(hidden_dim * 2)
        self.self_attn = MultiheadAttention(embed_dim=self.attn_dim, num_heads=4, batch_first=True)
        self.attn_layer_norm = LayerNorm(self.attn_dim)
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + fp_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(dropout_rate),
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def _node_embed(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gin_bn(self.gin(x, edge_index))
        x = self.ggnn_bn(self.ggnn(x, edge_index))
        return x, batch

    def _apply_attn(self, x, batch):
        batch_size = int(batch.max().item()) + 1
        sizes = torch.bincount(batch, minlength=batch_size)
        max_size = int(sizes.max().item())
        x_pad = x.new_zeros(batch_size, max_size, self.attn_dim)
        pad_mask = torch.ones(batch_size, max_size, dtype=torch.bool, device=x.device)
        for i in range(batch_size):
            idx = (batch == i).nonzero(as_tuple=True)[0]
            n = idx.numel()
            x_pad[i, :n] = x[idx]
            pad_mask[i, :n] = False
        attn_out, attn_w = self.self_attn(
            x_pad, x_pad, x_pad, key_padding_mask=pad_mask)
        attn_out = self.attn_layer_norm(attn_out + x_pad)
        attended, weights = [], []
        for i in range(batch_size):
            n = int((~pad_mask[i]).sum().item())
            attended.append(attn_out[i, :n])
            weights.append(attn_w[i, :n, :n])
        return torch.cat(attended, dim=0), weights

    def forward(self, data, fp):
        x, batch = self._node_embed(data)
        attended, _ = self._apply_attn(x, batch)
        pooled = global_add_pool(attended, batch)
        return self.fc(self.shared_fc(torch.cat((pooled, fp), dim=-1))).squeeze()

    def atom_attention_weights(self, data):
        x, batch = self._node_embed(data)
        assert int(batch.max().item()) == 0
        _, weights = self._apply_attn(x, batch)
        attn = weights[0]
        attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)
        atom_weights = attn.mean(dim=0)
        return atom_weights - atom_weights.mean()
