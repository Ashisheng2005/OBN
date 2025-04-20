#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2025/4/17 下午8:13 
# @Author : Huzhaojun
# @Version：V 1.0
# @File : scaler.py
# @desc : README.md

from sklearn.preprocessing import StandardScaler
from joblib import dump
from src.utils.config import Config
from src.utils.logger import setup_logger


class DataScaler:
    """管理数据标准化和扩展"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger(config)
        self.scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def fit_transform(self, X, y=None, is_regression=False):
        X_scaled = self.scaler.fit_transform(X)
        if is_regression and y is not None:
            y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
            return X_scaled, y_scaled
        return X_scaled, y

    def transform(self, X, y=None, is_regression=False):
        X_scaled = self.scaler.transform(X)
        if is_regression and y is not None:
            y_scaled = self.y_scaler.transform(y.reshape(-1, 1)).flatten()
            return X_scaled, y_scaled
        return X_scaled, y

    def save(self, model_type):
        model_dir = self.config.get_nested('output', 'model_dir', default='./model/')
        dump(self.scaler, f"{model_dir}scaler_{model_type}.pkl")
        if model_type == 'regression':
            dump(self.y_scaler, f"{model_dir}y_scaler_regression.pkl")
