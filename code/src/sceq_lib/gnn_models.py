"""
GNN-cGAN 网络结构定义 — 必须与训练代码完全一致
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from .config import cfg


class GeneratorNormal(nn.Module):
    """Normal 生成器: 月份条件 + 自回归 (日级递推)"""

    def __init__(self):
        super().__init__()
        self.cond_proj_layer = nn.Linear(cfg.COND_FEATURE_DIM, cfg.CLASS_EMBED_DIM)
        self.init_proj = nn.Linear(
            cfg.LATENT_DIM + cfg.CLASS_EMBED_DIM, cfg.N_NODES * cfg.HIDDEN_DIM
        )
        self.input_embedding = nn.Linear(cfg.N_FEATURES, cfg.EMBED_DIM)
        self.gat = GATConv(cfg.EMBED_DIM, cfg.HIDDEN_DIM, heads=2, concat=False)
        self.gru_cell = nn.GRUCell(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM)
        self.output_proj = nn.Linear(cfg.HIDDEN_DIM, cfg.N_FEATURES)
        self.act = nn.Tanh()

    def forward(self, z, edge_index, c, x_prev):
        """
        z:       (bs, 64)   噪声
        edge_index: (2, E)  图结构
        c:       (bs, 8)    月份条件
        x_prev:  (bs, 222, 4) 前一天最后一小时
        returns: (bs, 24, 222, 4)
        """
        bs = z.shape[0]
        h = self.init_proj(
            torch.cat([z, self.cond_proj_layer(c)], dim=1)
        ).view(bs * cfg.N_NODES, cfg.HIDDEN_DIM)
        curr_x = x_prev.view(bs * cfg.N_NODES, cfg.N_FEATURES)
        outs = []
        edge_idx_b = edge_index.repeat(1, bs) + torch.arange(
            bs, device=z.device
        ).repeat_interleave(edge_index.size(1)) * cfg.N_NODES
        for t in range(cfg.SEQ_LEN):
            x_emb = torch.relu(self.input_embedding(curr_x))
            h_spat = torch.relu(self.gat(x_emb, edge_idx_b))
            h = self.gru_cell(h_spat, h)
            out = self.output_proj(h)
            outs.append(out)
            curr_x = out
        return self.act(
            torch.stack(outs, 0)
            .view(cfg.SEQ_LEN, bs, cfg.N_NODES, cfg.N_FEATURES)
            .permute(1, 0, 2, 3)
        )


class GeneratorExtreme(nn.Module):
    """Extreme 生成器: 极端类型条件 + 冷启动 (独立 24h 切片)"""

    def __init__(self, num_classes=4):
        super().__init__()
        self.class_embedding = nn.Embedding(num_classes, cfg.CLASS_EMBED_DIM)
        self.init_proj = nn.Linear(
            cfg.LATENT_DIM + cfg.CLASS_EMBED_DIM, cfg.N_NODES * cfg.HIDDEN_DIM
        )
        self.input_embedding = nn.Linear(cfg.N_FEATURES, cfg.EMBED_DIM)
        self.gat = GATConv(cfg.EMBED_DIM, cfg.HIDDEN_DIM, heads=2, concat=False)
        self.gru_cell = nn.GRUCell(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM)
        self.output_proj = nn.Linear(cfg.HIDDEN_DIM, cfg.N_FEATURES)
        self.act = nn.Tanh()

    def forward(self, z, edge_index, c):
        """
        z:       (bs, 64)   噪声
        edge_index: (2, E)  图结构
        c:       (bs,)      极端类型标签 (1/2/3)
        returns: (bs, 24, 222, 4)
        """
        bs = z.shape[0]
        h = self.init_proj(
            torch.cat([z, self.class_embedding(c)], dim=1)
        ).view(bs * cfg.N_NODES, cfg.HIDDEN_DIM)
        curr_x = torch.zeros(bs * cfg.N_NODES, cfg.N_FEATURES).to(z.device)
        outs = []
        edge_idx_b = edge_index.repeat(1, bs) + torch.arange(
            bs, device=z.device
        ).repeat_interleave(edge_index.size(1)) * cfg.N_NODES
        for t in range(cfg.SEQ_LEN):
            x_emb = torch.relu(self.input_embedding(curr_x))
            h_spat = torch.relu(self.gat(x_emb, edge_idx_b))
            h = self.gru_cell(h_spat, h)
            out = self.output_proj(h)
            outs.append(out)
            curr_x = out
        return self.act(
            torch.stack(outs, 0)
            .view(cfg.SEQ_LEN, bs, cfg.N_NODES, cfg.N_FEATURES)
            .permute(1, 0, 2, 3)
        )
