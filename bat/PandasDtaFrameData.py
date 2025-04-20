#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2025/4/11 下午5:53 
# @Author : Huzhaojun
# @Version：V 1.0
# @File : PandasDtaFrameData.py
# @desc : README.md

import pandas as pd
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

class PandasDtaFrameData:

    def __init__(self, data):
        # 示例数据
        self.data = data

        # self.data = {
        #     "as_path_length": [2, 2, 1, 3, 1],  # AS_PATH 长度
        #     "local_pref": [100, 100, 100, 80, 90],  # BGP LOCAL_PREF
        #     "ospf_state": [1, 1, 0, 1, 1],  # OSPF 状态（1=FULL, 0=DOWN）
        #     "bandwidth": [1000, 1000, 500, 2000, 1000],  # 链路带宽（Mbps）
        #     "is_best": [1, 0, 1, 0, 1]  # 目标变量：是否最佳路径（1=True, 0=False）
        #     } if data.empty() else data
        # DataFrame 无法实现全局的bool评估，只能使用empty()检测是否为空

        self.df = pd.DataFrame(self.data)
        # 添加对数变化 bandwidth范围大（90-10959），可能影响模型对特征权重的学习。
        self.df["bandwidth_log"] = np.log1p(self.df["bandwidth"])
        logging.basicConfig(level=logging.INFO)

    def table_classification_model(self):
        """分类模型使用的数据标签,分类模型（输出 sigmoid）无法处理非0/1的标签"""
        # 特征 (X) 和目标变量 (y)
        X = self.df.drop(["is_best", "bandwidth"], axis=1).values  # 使用bandwidth_log
        y = self.df["is_best"].values

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,  # 保持类别比例
            random_state=42  # 可选：固定种子便于复现
        )

        # 数据标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)  # 只调用transform，确保尺度一致

        # 保存Scaler
        dump(scaler, "./model/scaler_classification.pkl")

        # 验证数据分布
        logging.info(f"X_train shape: {X_train.shape}, y_train classes: {np.unique(y_train, return_counts=True)}")
        logging.info(f"X_test shape: {X_test.shape}, y_test classes: {np.unique(y_test, return_counts=True)}")

        return X_train, X_test, y_train, y_test

    def table_regression_model(self):
        """回归模型数据，使用path_cost作为目标"""
        if "path_cost" not in self.df.columns:
            raise ValueError("Regression requires continuous target 'path_cost', but not found.")

        X = self.df.drop(["is_best", "path_cost"], axis=1).values
        y = self.df["path_cost"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

        dump(scaler, "./model/scaler_regression.pkl")
        dump(y_scaler, "./model/y_scaler_regression.pkl")

        logging.info(f"Regression data: y_train mean={y_train.mean():.4f}, std={y_train.std():.4f}")

        return X_train, X_test, y_train, y_test
