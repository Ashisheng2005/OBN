#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2025/4/17 下午8:13 
# @Author : Huzhaojun
# @Version：V 1.0
# @File : classification.py
# @desc : README.md

"""
定义分类模型（TensorFlow神经网络），预测is_best，使用binary_crossentropy损失和sigmoid输出。
"""

import tensorflow as tf
from sklearn.metrics import precision_recall_curve
import numpy as np
from src.utils.config import Config
from src.utils.logger import setup_logger


class ClassificationModel:
    """定义分类模型"""

    def __init__(self, config: Config, input_shape):
        self.config = config
        self.logger = setup_logger(config)
        self.model = None
        self.input_shape = input_shape
        self.build()

    def build(self):
        data_size = self.input_shape[0]
        capacity_unit = (
            self.config.get_nested('model', 'classification', 'capacity_unit', 'large') if data_size > 4500 else
            self.config.get_nested('model', 'classification', 'capacity_unit', 'medium') if data_size > 950 else
            self.config.get_nested('model', 'classification', 'capacity_unit', 'small')
        )
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                capacity_unit, activation='relu', input_shape=(self.input_shape[1],),
                kernel_regularizer=tf.keras.regularizers.l2(0.02)
            ),
            tf.keras.layers.Dropout(0.7),
            tf.keras.layers.Dense(
                capacity_unit // 2, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.02)
            ),
            tf.keras.layers.Dropout(0.6),
            tf.keras.layers.Dense(capacity_unit // 4, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

    def train(self, X_train, y_train, X_val, y_val, X_test, y_test):
        # 获取自定义配置
        epochs = self.config.get_nested('model', 'classification', 'epochs', default=200)
        batch_size = self.config.get_nested('model', 'classification', 'batch_size', default=32)

        # 调整类权重，基于原始比例
        pos_weight = (len(y_train) / np.sum(y_train == 1)) * 0.5  # 原始比例约为16.36，乘以0.5以避免过大
        class_weights = {0: 1.0, 1: pos_weight}
        self.logger.info(f"Class weights: {class_weights}")

        # 早停
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

        history = self.model.fit(
            X_train, y_train,  # 移除 tf.expand_dims
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size,
            class_weight=class_weights,  # 添加类权重
            callbacks=[early_stopping, lr_scheduler], verbose=1
        )

        # 评估验证集和测试集
        loss, accuracy, precision, recall = self.model.evaluate(X_val, y_val)
        self.logger.info(f"Validation: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

        loss, accuracy, precision, recall = self.model.evaluate(X_test, y_test)
        self.logger.info(f"Test: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

        # 动态阈值优化
        y_val_pred = self.model.predict(X_val)
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_pred)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

        # 加权 F1 分数，增加对精确率的权重
        weights = precisions  # 更关注高精确率
        weighted_f1 = f1_scores * weights / (weights.sum() + 1e-10)
        optimal_threshold = thresholds[np.argmax(weighted_f1)]
        self.logger.info(
            f"Optimal threshold (weighted F1): {optimal_threshold:.4f}, Weighted F1: {np.max(weighted_f1):.4f}")

        y_test_pred = (self.model.predict(X_test) >= optimal_threshold).astype(int)
        accuracy = np.mean(y_test_pred == y_test)
        self.logger.info(f"Test Accuracy with optimal threshold: {accuracy:.4f}")

        # 单独评估正类
        # pos_mask = y_test == 1
        # if pos_mask.sum() > 0:
        #     pos_precision = precision_score(y_test[pos_mask], y_test_pred[pos_mask])
        #     pos_recall = recall_score(y_test[pos_mask], y_test_pred[pos_mask])
        #     pos_f1 = f1_score(y_test[pos_mask], y_test_pred[pos_mask])
        #     self.logger.info(
        #         f"Positive class (is_best=1): Precision={pos_precision:.4f}, Recall={pos_recall:.4f}, F1={pos_f1:.4f}")

        return history, accuracy

    def save_model(self):
        model_dir = self.config.get_nested('output', 'model_dir', default='./model/')
        # self.model.save_weights(f"{model_dir}best_path_classification_model.weights.h5")
        # self.model.load_weights(f"{model_dir}best_path_classification_model.weights.h5")
        self.model.save(f"{model_dir}best_path_classification_model.weights.keras")

