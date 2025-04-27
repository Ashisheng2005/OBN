#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2025/4/26 下午11:47 
# @Author : Huzhaojun
# @Version：V 1.0
# @File : gnn_model.py
# @desc : README.md

import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from src.utils.config import Config
from src.utils.logger import setup_logger
from sklearn.metrics import f1_score, recall_score


class GNNModel(nn.Module):
    def __init__(self, config: Config, input_dim, hidden_dim=64):
        super(GNNModel, self).__init__()
        self.config = config
        self.logger = setup_logger(config)
        self.conv1 = pyg_nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, hidden_dim // 2)
        self.fc = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.fc(x)
        return torch.sigmoid(x)

    def train_model(self, data, epochs=40, lr=0.01):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()

        # 添加早停机制
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        best_model_state = None

        # 分割数据（假设 data 包含训练和验证集）
        train_mask = data.train_mask
        val_mask = data.val_mask
        y_true = data.y

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            out = self(data)
            train_loss = criterion(out[train_mask], y_true[train_mask])
            train_loss.backward()
            optimizer.step()

            # 验证集评估
            self.eval()
            with torch.no_grad():
                val_out = self(data)
                val_loss = criterion(val_out[val_mask], y_true[val_mask])

                # 计算 F1 分数和 Recall
                val_pred = (val_out[val_mask] >= 0.5).float()
                val_true = y_true[val_mask]
                f1 = f1_score(val_true.cpu().numpy(), val_pred.cpu().numpy())
                recall = recall_score(val_true.cpu().numpy(), val_pred.cpu().numpy())

            self.logger.info(
                f"Epoch {epoch + 1}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val F1: {f1:.4f}, Val Recall: {recall:.4f}")

            # 早停逻辑
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # 恢复最佳模型
        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        return self