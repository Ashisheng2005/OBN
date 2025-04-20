#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2025/4/17 下午8:13 
# @Author : Huzhaojun
# @Version：V 1.0
# @File : classification.py
# @desc : README.md

import tensorflow as tf
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
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            ),
            tf.keras.layers.Dropout(0.6),
            tf.keras.layers.Dense(
                capacity_unit // 2, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(capacity_unit // 4, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train, y_train, X_test, y_test):
        epochs = self.config.get_nested('model', 'classification', 'epochs', default=200)
        batch_size = self.config.get_nested('model', 'classification', 'batch_size', default=32)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        history = self.model.fit(
            # X_train
            tf.expand_dims(X_train, axis=-1), y_train, validation_data=(X_test, y_test),
            epochs=epochs, batch_size=batch_size, callbacks=[early_stopping, lr_scheduler], verbose=1
        )
        loss, accuracy = self.model.evaluate(X_test, y_test)
        self.logger.info(f"Test Accuracy: {accuracy * 100:.2f}%")
        return history, accuracy

    def save_model(self):
        model_dir = self.config.get_nested('output', 'model_dir', default='./model/')
        # self.model.save_weights(f"{model_dir}best_path_classification_model.weights.h5")
        # self.model.load_weights(f"{model_dir}best_path_classification_model.weights.h5")
        self.model.save(f"{model_dir}best_path_classification_model.weights.keras")

