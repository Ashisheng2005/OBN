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
from sklearn.metrics import f1_score, recall_score
import numpy as np
from src.utils.config import Config
from src.utils.logger import setup_logger
import optuna


class ClassificationModel:
    """定义分类模型"""

    def __init__(self, config: Config, input_shape, trial=None):
        self.config = config
        self.logger = setup_logger(config)
        self.model = None
        self.input_shape = input_shape
        self.trial = trial
        self.build()

    def build(self):
        data_size = self.input_shape[0]
        n_layers = self.trial.suggest_int('n_layers', 2, 5) if self.trial else 3
        capacity_unit = (
            self.config.get_nested('model', 'classification', 'capacity_unit', 'large') if data_size > 4500 else
            self.config.get_nested('model', 'classification', 'capacity_unit', 'medium') if data_size > 950 else
            self.config.get_nested('model', 'classification', 'capacity_unit', 'small')
        )
        dropout_rate = self.trial.suggest_float('dropout_rate', 0.2, 0.4) if self.trial else 0.3
        l2_strength = self.trial.suggest_float('l2_strength', 1e-4, 1e-2, log=True) if self.trial else 0.01

        inputs = tf.keras.layers.Input(shape=(self.input_shape[1],))
        x = inputs
        for i in range(n_layers):
            units = capacity_unit // (2 ** i)
            x = tf.keras.layers.Dense(
                units, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            if i % 2 == 0 and i > 0:
                shortcut = tf.keras.layers.Dense(units, use_bias=False)(inputs)
                x = tf.keras.layers.Add()([x, shortcut])
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        self.model = tf.keras.Model(inputs, outputs)

        learning_rate = self.trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True) if self.trial else 0.0005
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
        # 尝试使用 tf.keras.optimizers.Adam 替代 AdamW，可能加速收敛
        # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        # data_size = self.input_shape[0]
        # capacity_unit = (
        #     self.config.get_nested('model', 'classification', 'capacity_unit', 'large') if data_size > 4500 else
        #     self.config.get_nested('model', 'classification', 'capacity_unit', 'medium') if data_size > 950 else
        #     self.config.get_nested('model', 'classification', 'capacity_unit', 'small')
        # )
        # self.model = tf.keras.Sequential([
        #     tf.keras.layers.Dense(
        #         capacity_unit, activation='relu', input_shape=(self.input_shape[1],),
        #         kernel_regularizer=tf.keras.regularizers.l2(0.02)
        #     ),
        #     tf.keras.layers.Dropout(0.7),
        #     tf.keras.layers.Dense(
        #         capacity_unit // 2, activation='relu',
        #         kernel_regularizer=tf.keras.regularizers.l2(0.02)
        #     ),
        #     tf.keras.layers.Dropout(0.6),
        #     tf.keras.layers.Dense(capacity_unit // 4, activation='relu'),
        #     tf.keras.layers.Dense(1, activation='sigmoid')
        # ])
        # self.model.compile(
        #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        #     loss='binary_crossentropy',
        #     metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        # )

    # def train(self, X_train, y_train, X_val, y_val, X_test, y_test, class_weights):
    #     # 获取自定义配置
    #     epochs = self.config.get_nested('model', 'classification', 'epochs', default=200)
    #     batch_size = self.config.get_nested('model', 'classification', 'batch_size', default=32)
    #
    #     # 调整类权重，基于原始比例
    #     # pos_weight = (len(y_train) / np.sum(y_train == 1)) * 0.5  # 原始比例约为16.36，乘以0.5以避免过大
    #     # class_weights = {0: 1.0, 1: pos_weight}
    #     self.logger.info(f"Class weights: {class_weights}")
    #
    #     # 早停
    #     early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    #     lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    #
    #     history = self.model.fit(
    #         X_train, y_train,  # 移除 tf.expand_dims
    #         validation_data=(X_val, y_val),
    #         epochs=epochs, batch_size=batch_size,
    #         class_weight=class_weights,  # 添加类权重
    #         callbacks=[early_stopping, lr_scheduler], verbose=1
    #     )
    #
    #     # 评估验证集和测试集
    #     loss, accuracy, precision, recall = self.model.evaluate(X_val, y_val)
    #     self.logger.info(f"Validation: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    #
    #     loss, accuracy, precision, recall = self.model.evaluate(X_test, y_test)
    #     self.logger.info(f"Test: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    #
    #     # 动态阈值优化
    #     y_val_pred = self.model.predict(X_val)
    #     precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_pred)
    #     f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    #
    #     # 更关注少数类的F1分数
    #     optimal_idx = np.argmax(f1_scores)
    #     optimal_threshold = thresholds[optimal_idx]
    #     self.logger.info(f"Optimal threshold (max F1): {optimal_threshold:.4f}, F1: {f1_scores[optimal_idx]:.4f}")
    #
    #     # 评估测试集
    #     y_test_pred = (self.model.predict(X_test) >= optimal_threshold).astype(int)
    #     accuracy = np.mean(y_test_pred == y_test)
    #     self.logger.info(f"Test Accuracy with optimal threshold: {accuracy:.4f}")
    #
    #     return history, accuracy

    def train(self, X_train, y_train, X_val, y_val, X_test, y_test, class_weights):
        epochs = self.config.get_nested('model', 'classification', 'epochs', default=100)
        batch_size = self.trial.suggest_int('batch_size', 16, 64) if self.trial else 32

        # 调整类权重，增强对少数类的关注
        pos_weight = (len(y_train) / (np.sum(y_train == 1) + 1e-10)) * 0.8  # 动态计算权重
        class_weights = {0: 1.0, 1: pos_weight}
        self.logger.info(f"Adjusted class weights: {class_weights}")

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
        warmup_scheduler = tf.keras.callbacks.LambdaCallback(
            on_epoch_begin=lambda epoch, logs: self.model.optimizer.learning_rate.assign(
                min(1.0, (epoch + 1) / 3.0) * self.model.optimizer.learning_rate.numpy()
            )
        )

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size,
            class_weight=class_weights,
            callbacks=[early_stopping, lr_scheduler, warmup_scheduler],
            verbose=1
        )

        self.logger.info("Training completed")

        # 评估
        loss, accuracy, precision, recall = self.model.evaluate(X_val, y_val)
        self.logger.info(f"Validation: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
        loss, accuracy, precision, recall = self.model.evaluate(X_test, y_test)
        self.logger.info(f"Test: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

        # 动态阈值优化
        y_val_pred = self.model.predict(X_val)
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_pred)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        self.logger.info(f"Optimal threshold (max F1): {optimal_threshold:.4f}, F1: {f1_scores[optimal_idx]:.4f}")

        y_test_pred = (self.model.predict(X_test) >= optimal_threshold).astype(int)
        accuracy = np.mean(y_test_pred == y_test)
        self.logger.info(f"Test Accuracy with optimal threshold: {accuracy:.4f}")

        # 对少数类的f1分数和Recall评估
        # y_test_pred = (self.model.predict(X_test) >= optimal_threshold).astype(int)
        f1 = f1_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        self.logger.info(f"Test F1 Score: {f1:.4f}, Recall: {recall:.4f}")

        return history, accuracy

    def save_model(self):
        model_dir = self.config.get_nested('output', 'model_dir', default='./model/')
        # self.model.save_weights(f"{model_dir}best_path_classification_model.weights.h5")
        # self.model.load_weights(f"{model_dir}best_path_classification_model.weights.h5")
        self.model.save(f"{model_dir}best_path_classification_model.weights.keras")

