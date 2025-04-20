#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2025/4/17 下午8:14 
# @Author : Huzhaojun
# @Version：V 1.0
# @File : plotter.py
# @desc : README.md

import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from src.utils.config import Config
from src.utils.logger import setup_logger


class Plotter:
    """处理培训历史和指标的绘图"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger(config)
        self.plot_dir = self.config.get_nested('output', 'plot_dir', default='./chart/')

    def plot_edge_cases(self, df):
        plt.figure(figsize=(10, 6))
        for interface in ['FastEthernet', 'GigabitEthernet', 'TenGigabitEthernet']:
            mask = df[f'is_{interface}'] == 1
            plt.hist(df[mask]['packet_loss'], bins=50, alpha=0.5, label=interface)
        plt.yscale('log')
        plt.title('Packet Loss Distribution by Interface Type (Log Scale)')
        plt.xlabel('Packet Loss')
        plt.ylabel('Frequency')

        plt.legend()
        # plt.show()
        plt.savefig(f"{self.plot_dir}Packet_Loss_Distribution.png")
        plt.close()

    def plot_pr_roc(self, y_true, y_pred_proba, model_type):
        plt.figure(figsize=(12, 4))

        # PR 曲线
        precisions, recalls, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = np.trapz(recalls, precisions)
        plt.subplot(1, 2, 1)
        plt.plot(recalls, precisions, label=f'PR-AUC={pr_auc:.4f}')
        plt.title(f'{model_type} PR Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()

        # ROC 曲线
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = np.trapz(tpr, fpr)
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, label=f'ROC-AUC={roc_auc:.4f}')
        plt.title(f'{model_type} ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()

        plt.savefig(f"{self.plot_dir}{model_type}_pr_roc.png")
        plt.close()
        self.logger.info(f"Saved PR/ROC plot to {self.plot_dir}{model_type}_pr_roc.png")

    def plot_edge_boxplot(self, df):
        plt.figure(figsize=(10, 6))
        df.boxplot(column='packet_loss', by=['is_FastEthernet', 'is_GigabitEthernet', 'is_TenGigabitEthernet'])
        plt.title('Packet Loss Boxplot by Interface Type')
        plt.xlabel('Interface Type')
        plt.ylabel('Packet Loss')
        # plt.show()
        plt.savefig(f"{self.plot_dir}Packet_Loss_Boxplot.png")
        plt.close()

    def plot_training_history(self, history, model_name, data_size, accuracy=None, r2=None):
        # plot_dir = self.config.get_nested('output', 'plot_dir', default='./chart/')
        os.makedirs(self.plot_dir, exist_ok=True)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title(f'{model_name} Model Loss [data size: {data_size}]')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        if 'accuracy' in history.history:
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Val Accuracy')
            plt.title(f'{model_name} Model Accuracy [Accuracy: {accuracy:.4f}]')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
        else:
            plt.plot(history.history['mae'], label='Train MAE')
            plt.plot(history.history['val_mae'], label='Val MAE')
            plt.title(f'{model_name} Model MAE [R2: {r2:.4f}]')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
        plt.legend()

        plt.savefig(f"{self.plot_dir}{model_name}_{data_size}.png")
        plt.close()
        self.logger.info(f"Saved plot to {self.plot_dir}{model_name}_{data_size}.png")
