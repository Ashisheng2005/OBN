#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2025/4/11 下午11:11 
# @Author : Huzhaojun
# @Version：V 1.0
# @File : start.py
# @desc : README.md

import tensorflow as tf
from joblib import load
import numpy as np
from Collect_network_data import Collect_network_data

# 加载分类模型（预测最佳路径：0/1）
clf_model = tf.keras.models.load_model(r'.\model\best_path_classification_model.weights.h5')

# 加载回归模型（预测延迟：连续值）
reg_model = tf.keras.models.load_model(r'.\model\best_path_regression_model.weights.h5')

# 加载训练时的scaler
scaler = load('./model/scaler.pkl')

# 将实时数据转换为模型输入格式
ospf, bgp = Collect_network_data(mode="real").main()

# 提取关键特征（需与训练时完全一致！）
features = {
        'as_path_length': len(bgp[0]['as_path'].split()),
        'local_pref': bgp[0]['local_pref'],
        'ospf_state': 1 if ospf[0]['state'] == 'FULL' else 0,
        'bandwidth': 1000  # 假设固定值或从接口数据获取
}

X_new = np.array([[features['as_path_length'],
                 [features['local_pref']],
                 [features['ospf_state']],
                 [features['bandwidth']]]]
                 )  # 形状需匹配训练数据

X_new = scaler.transform(X_new)

# 分类预测
best_path_prob = clf_model.predict(X_new)[0][0]  # 输出概率
best_path = "Yes" if best_path_prob > 0.5 else "No"
print(f"Should choose this path? {best_path} (Confidence: {best_path_prob:.2f})")

# 回归预测(路径延迟)
pred_delay = reg_model.predict(X_new)[0][0]
print(f"Predicted delay: {pred_delay:.2f} ms")