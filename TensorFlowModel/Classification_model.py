#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2025/4/11 下午5:55 
# @Author : Huzhaojun
# @Version：V 1.0
# @File : Classification_model.py
# @desc : README.md

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import r2_score
import logging


class Model:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.classification_model = None
        self.regression_model = None
        # 设置日志级别
        logging.basicConfig(level=logging.INFO)

    def Classification_model(self, epochs: int = 100, batch_size: int = 32, data_size: int = None):
        # 动态容量单元
        capacity_unit = 512 if data_size > 4500 else 256 if data_size > 950 else 128

        # 构建更深的分类模型
        self.classification_model = tf.keras.models.Sequential([
            # 添加L2正则化（kernel_regularizer），减少过拟合。
            Dense(capacity_unit,
                  activation="relu",
                  input_shape=(self.X_train.shape[1],),
                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dropout(0.6),   # 防止过拟合
            Dense(capacity_unit // 2,
                  activation="relu",
                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dropout(0.5),   # 防止过拟合
            Dense(capacity_unit // 4,
                  activation="relu"),
            Dense(1, activation="sigmoid")  # 输出层（二分类）
        ])

        # 编译模型
        self.classification_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        # 设置早停和学习率调度
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

        # 训练模型
        history = self.classification_model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, lr_scheduler],
            verbose=1
        )

        # 评估模型
        loss, accuracy = self.classification_model.evaluate(self.X_test, self.y_test)
        logging.info(f"Test Accuracy: {accuracy * 100:.2f}%")

        # 保存权重
        self.classification_model.save_weights("./model/best_path_classification_model.weights.h5")    # 保存为 HDF5 文件
        return history, accuracy

    def Regression_model(self, epochs: int = 75, batch_size: int = 32, data_size: int = None):
        capacity_unit = 512 if data_size > 4500 else 256 if data_size > 950 else 128

        # 构建回归模型 增加Dropout和L2正则化
        self.regression_model = tf.keras.models.Sequential([
            Dense(capacity_unit,
                  activation="relu",
                  input_shape=(self.X_train.shape[1],),
                  kernel_regularizer=tf.keras.regularizers.l2(0.01)
                  ),

            Dropout(0.4),
            Dense(capacity_unit // 2,
                  activation="relu",
                  kernel_regularizer=tf.keras.regularizers.l2(0.01)
                  ),

            Dropout(0.3),
            Dense(capacity_unit // 4, activation="relu"),
            Dense(1)  # 输出层（无激活函数）
        ])

        # 编译模型
        self.regression_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )

        # 设置早停和学习率调度
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

        # 训练模型
        history = self.regression_model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, lr_scheduler],
            verbose=1
        )

        # 评估模型
        loss, mae = self.regression_model.evaluate(self.X_test, self.y_test)
        logging.info(f"Test MSE: {loss:.4f}, Test MAE: {mae:.4f}")

        # R²分数衡量模型解释目标方差的能力，值越接近1越好。
        y_pred = self.regression_model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        logging.info(f"Test R² Score: {r2:.4f}")

        # 保存权重为 HDF5 文件
        self.regression_model.save_weights("./model/best_path_regression_model.weights.h5")
        return history, r2

