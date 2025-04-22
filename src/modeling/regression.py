#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2025/4/17 下午8:14 
# @Author : Huzhaojun
# @Version：V 1.0
# @File : regression.py
# @desc : README.md

"""
定义回归模型（TensorFlow神经网络），预测path_cost，使用mse损失。
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score
from src.utils.config import Config
from src.utils.logger import setup_logger


class RegressionModel:
    """定义回归模型"""

    def __init__(self, config: Config, input_shape):
        self.config = config
        self.logger = setup_logger(config)
        self.model = None
        self.input_shape = input_shape
        self.build()

    def build(self):
        data_size = self.input_shape[0]
        capacity_unit = (
            self.config.get_nested('model', 'regression', 'capacity_unit', 'large') if data_size > 4500 else
            self.config.get_nested('model', 'regression', 'capacity_unit', 'medium') if data_size > 950 else
            self.config.get_nested('model', 'regression', 'capacity_unit', 'small')
        )
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_shape[1],)),
            tf.keras.layers.Dense(
                capacity_unit, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.05)
            ),
            tf.keras.layers.Dropout(0.6),
            tf.keras.layers.Dense(
                capacity_unit // 2, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.05)
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(capacity_unit // 4, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
            loss='mse',
            metrics=['mae']
        )

    def train(self, X_train, y_train, X_val, y_val, X_test, y_test, weights_train=None, y_test_is_best=None):
        """训练函数"""

        # 获取具体配置，若无则使用默认配置
        epochs = self.config.get_nested('model', 'regression', 'epochs', default=200)
        batch_size = self.config.get_nested('model', 'regression', 'batch_size', default=32)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

        # 检查 weights_train 是否为 None
        if weights_train is not None:
            weights_train = np.where(weights_train > 1, 1.2, 1.0)  # 降低is_best=1权重

        else:
            self.logger.warning("weights_train is None, using uniform weights.")
            weights_train = np.ones(len(y_train))  # 使用均匀权重

        # 添加噪声鲁棒性：对训练数据添加小幅噪声
        X_train_noisy = X_train + np.random.normal(0, 0.01, X_train.shape)

        history = self.model.fit(
            X_train_noisy, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size,
            sample_weight=weights_train,
            callbacks=[early_stopping, lr_scheduler], verbose=1
        )

        # loss, mae = self.model.evaluate(X_test, y_test)
        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        self.logger.info(f"Test R2: {r2:.4f}")

        # 评估is_best=1样本
        minority_mask = y_test == 1  # 假设可以访问原始y_test的is_best标签
        if minority_mask.sum() > 0:
            minority_r2 = r2_score(y_test[minority_mask], y_pred[minority_mask])
            self.logger.info(f"R2 for minority class (is_best=1): {minority_r2:.4f}")

        return history, r2

    def save_model(self):
        model_dir = self.config.get_nested('output', 'model_dir', default='./model/')
        # self.model.save_weights(f"{model_dir}best_path_regression_model.weights.h5")
        self.model.save(f"{model_dir}best_path_regression_model.weights.keras")