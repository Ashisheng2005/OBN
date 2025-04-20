#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2025/4/17 下午8:15 
# @Author : Huzhaojun
# @Version：V 1.0
# @File : test_collector.py
# @desc : README.md


from keras.models import load_model
from joblib import load
import numpy as np
import pandas as pd
from src.utils.config import Config
from sklearn.metrics import accuracy_score, f1_score
from os import getcwd
from src.utils.logger import setup_logger


class TextCollector:

    def __init__(self, config: Config=None):
        # self.config = config
        # self.logger = setup_logger(self.config)
        # self.collector_model_path = self.config.get_nested('output', 'model_dir')
        self.collector_model_path = "../model/"

        # 加载分类模型 预测最佳路径 0/1
        self.clf_model = load_model(self.collector_model_path + "best_path_classification_model.weights.keras")
        print(self.clf_model.input_shape)
        # 加载训练时的scaler
        self.scaler = load(self.collector_model_path + "scaler_classification.pkl")

    def text(self):
        # 定义训练时使用的特征
        required_features = [
            "as_path_length", "local_pref", "ospf_state", "bandwidth",
            "latency", "packet_loss", "bandwidth_utilization", "jitter",
            "rtt", "cpu_usage", "memory_usage", "queue_length",
            "route_stability", "latency_bandwidth_ratio",
            "is_FastEthernet", "is_GigabitEthernet", "is_TenGigabitEthernet",
            "bandwidth_log", "jitter_log", "rtt_log", "resource_usage",
            "route_stability_log"
        ]

        # 准备新数据（示例：真实网络数据）
        raw_data = {
            "path_id": [101, 102],
            "as_path_length": [3, 4],
            "local_pref": [100, 120],
            "ospf_state": [1, 0],
            "bandwidth": [1000, 100],
            "latency": [15, 20],
            "packet_loss": [2.5, 10],
            "bandwidth_utilization": [50, 80],
            "jitter": [3, 5],
            "rtt": [30, 40],
            "cpu_usage": [20, 30],
            "memory_usage": [40, 50],
            "queue_length": [5, 10],
            "route_stability": [1, 2],
            "interface": ["GigabitEthernet", "FastEthernet"]
        }
        new_df = pd.DataFrame(raw_data)

        # 单热编码和衍生特征
        new_df["is_FastEthernet"] = (new_df["interface"] == "FastEthernet").astype(int)
        new_df["is_GigabitEthernet"] = (new_df["interface"] == "GigabitEthernet").astype(int)
        new_df["is_TenGigabitEthernet"] = (new_df["interface"] == "TenGigabitEthernet").astype(int)
        new_df = new_df.drop("interface", axis=1)

        new_df["latency_bandwidth_ratio"] = new_df["latency"] / new_df["bandwidth"]
        new_df["bandwidth_log"] = np.log1p(new_df["bandwidth"])
        new_df["jitter_log"] = np.log1p(new_df["jitter"])
        new_df["rtt_log"] = np.log1p(new_df["rtt"])
        new_df["resource_usage"] = 0.6 * new_df["cpu_usage"] + 0.4 * new_df["memory_usage"]
        new_df["route_stability_log"] = np.log1p(new_df["route_stability"])
        # new_df["latency_bandwidth_ratio_log"] = np.log1p(new_df["latency_bandwidth_ratio"])

        # 保留 path_id 用于追踪
        path_ids = new_df["path_id"]
        new_df = new_df.drop("path_id", axis=1)

        # 确保特征顺序
        new_df = new_df[required_features]
        for feature in required_features:
            if feature not in new_df.columns:
                new_df[feature] = 0  # 或其他默认值

        X_new = new_df.values

        # 预测
        predictions = self.clf_model.predict(X_new)
        predictions = (predictions > 0.5).astype(int)

        # 输出结果
        for path_id, pred in zip(path_ids, predictions):
            print(f"Path {path_id}: is_best={pred}")

        # 应用：选择最佳路径
        best_path = None
        for path_id, pred in zip(path_ids, predictions):
            if pred == 1:
                best_path = path_id
                print(f"Path {path_id} is selected as the best path")
                break

        if best_path is None:
            print("No best path found")

        # # 如果有真实标签，评估模型
        # y_true = [1, 0]  # 示例真实标签
        # if y_true:
        #     accuracy = accuracy_score(y_true, predictions)
        #     f1 = f1_score(y_true, predictions)
        #     print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")


if __name__ == '__main__':
    demo = TextCollector()
    demo.text()