#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2025/4/17 下午8:14 
# @Author : Huzhaojun
# @Version：V 1.0
# @File : trainer.py
# @desc : README.md

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.preprocessing.scaler import DataScaler
from src.modeling.classification import ClassificationModel
from src.modeling.regression import RegressionModel
from src.visualization.plotter import Plotter


class Trainer:
    """协调模型训练和数据预处理。"""

    def __init__(self, config: Config, data):
        self.config = config
        self.logger = setup_logger(config)
        self.data = data
        self.plotter = Plotter(config)

    def prepare_classification_data(self):
        """分类模型使用的数据标签,分类模型（输出 sigmoid）无法处理非0/1的标签"""

        # 特征 (X) 和目标变量 (y)
        X = self.data.drop(["is_best", "bandwidth", "path_cost"], axis=1).values
        y = self.data["is_best"].values

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2,
            stratify=y,             # 保持类别比例
            random_state=42         # 可选：固定种子便于复现
        )

        # 数据标准化
        scaler = DataScaler(self.config)
        X_train, y_train = scaler.fit_transform(X_train, y_train)
        X_test, y_test = scaler.transform(X_test, y_test)

        # 使用SMOTE 数据增强, 平衡is_best
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        scaler.save('classification')

        self.logger.info(f"Classification data: y_train distribution: {np.unique(y_train, return_counts=True)}")
        return X_train, X_test, y_train, y_test

    def prepare_regression_data(self):
        """回归模型数据，使用path_cost作为目标"""

        if "path_cost" not in self.data.columns:
            raise ValueError("Regression requires 'path_cost' column.")

        # 确保数据没有 NaN 值
        data_clean = self.data.dropna(subset=["is_best", "bandwidth", "path_cost"])
        self.logger.info(f"After dropping NaN: Data shape={data_clean.shape}")

        X = self.data.drop(["is_best", "bandwidth", "path_cost"], axis=1).values
        y = self.data["path_cost"].values

        self.logger.info(f"Raw data: X shape={X.shape}, y shape={y.shape}")
        # 检查是否存在 NaN 值
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            self.logger.warning(f"NaN values detected: X={np.isnan(X).sum()}, y={np.isnan(y).sum()}")

        # 确保 X 和 y 样本数一致
        if len(X) != len(y):
            self.logger.error(f"X and y mismatch: X={len(X)}, y={len(y)}")
            valid_indices = min(len(X), len(y))
            X = X[:valid_indices]
            y = y[:valid_indices]
            self.logger.info(f"Truncated to match: X shape={X.shape}, y shape={y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.logger.info(f"After split: X_train shape={X_train.shape}, y_train shape={y_train.shape}, "
                         f"X_test shape={X_test.shape}, y_test shape={y_test.shape}")

        scaler = DataScaler(self.config)
        X_train, y_train = scaler.fit_transform(X_train, y_train, is_regression=True)
        X_test, y_test = scaler.transform(X_test, y_test, is_regression=True)

        self.logger.info(f"After scaling: X_train shape={X_train.shape}, y_train shape={y_train.shape}")

        scaler.save('regression')
        self.logger.info(f"Regression data: y_train mean={y_train.mean():.4f}, std={y_train.std():.4f}")
        return X_train, X_test, y_train, y_test

    def train(self):
        # Classification
        try:
            self.logger.info("Training classification model")
            X_train, X_test, y_train, y_test = self.prepare_classification_data()
            model = ClassificationModel(self.config, X_train.shape)
            history, accuracy = model.train(X_train, y_train, X_test, y_test)
            model.save_model()
            self.plotter.plot_training_history(history, "Classification", len(X_train), accuracy=accuracy)
        except Exception as e:
            self.logger.error(f"Classification training error: {e}")
            raise

        # Regression
        try:
            self.logger.info("Training regression model")
            X_train, X_test, y_train, y_test = self.prepare_regression_data()
            model = RegressionModel(self.config, X_train.shape)
            history, r2 = model.train(X_train, y_train, X_test, y_test)
            model.save_model()
            self.plotter.plot_training_history(history, "Regression", len(X_train), r2=r2)
        except Exception as e:
            self.logger.error(f"Regression training error: {e}")
            raise
