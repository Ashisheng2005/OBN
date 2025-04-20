#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2025/4/17 下午8:14 
# @Author : Huzhaojun
# @Version：V 1.0
# @File : regression.py
# @desc : README.md

import tensorflow as tf
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
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            ),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(
                capacity_unit // 2, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            ),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(capacity_unit // 4, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

    def train(self, X_train, y_train, X_test, y_test):
        """训练函数"""

        # 获取具体配置，若无则使用默认配置
        epochs = self.config.get_nested('model', 'regression', 'epochs', default=200)
        batch_size = self.config.get_nested('model', 'regression', 'batch_size', default=32)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

        history = self.model.fit(
            # X_train,
            tf.expand_dims(X_train, axis=-1), y_train, validation_data=(X_test, y_test),
            epochs=epochs, batch_size=batch_size, callbacks=[early_stopping, lr_scheduler], verbose=1
        )

        # loss, mae = self.model.evaluate(X_test, y_test)
        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        return history, r2

    def save_model(self):
        model_dir = self.config.get_nested('output', 'model_dir', default='./model/')
        # self.model.save_weights(f"{model_dir}best_path_regression_model.weights.h5")
        self.model.save(f"{model_dir}best_path_regression_model.weights.keras")