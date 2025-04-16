#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2025/4/11 下午6:07 
# @Author : Huzhaojun
# @Version：V 1.0
# @File : Training_model.py
# @desc : README.md

from PandasDtaFrameData import PandasDtaFrameData
from TensorFlowModel.Classification_model import Model
from Collect_network_data import Collect_network_data
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from os.path import isfile
import numpy as np
from json import dumps, loads
import logging


def plot_training_history(history, model_name: str = "", data_size: int = None, r2: float = None):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Model Loss [data size: {data_size}]')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title(f'{model_name} Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
    else:
        plt.plot(history.history['mae'], label='Train MAE')
        plt.plot(history.history['val_mae'], label='Val MAE')
        plt.title(f'{model_name} Model MAE[R2: {r2:.4f}]')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
    plt.legend()
    plt.savefig(f"./chart/{model_name}_{data_size}.png")
    plt.show()


class run:

    def __init__(self, set: dict = None):
        logging.basicConfig(level=logging.DEBUG)
        self.set = {
            # 分类模型设置
            "Classification_model": {
                # 训练步数
                "epochs": 100
            },

            # 回归模型设置
            "Regression_model": {
                # 训练步数
                "epochs": 75,
            },

            # 真实环境设置
            "real": {
                # 链接设备的配置,支持多设备
                "device": [
                    {'device_type': 'cisco_ios',
                     'host': '192.168.1.1',
                     'username': 'admin',
                     'password': 'password'
                    },
                    {"device_type": "huawei",
                     "host": "router2",
                     "username": "admin",
                     "password": "password"
                     }
                ]
            },

            # 仿真环境设置
            "virtual": {
                # 生成的数据量
                "num_samples": 1000,

            }
        } if not set else set

        if not isfile("./config.json"):
            with open("./config.json", "w", encoding="utf-8") as f:
                f.write(dumps(self.set, indent=4, ensure_ascii=False))

    def main(self):

        training_data = Collect_network_data(mode_set=self.set).main()
        pd = PandasDtaFrameData(data=training_data)

        if isinstance(training_data, dict):
            training_data = pd.DataFrame(training_data)  # 确保为DataFrame

        logging.info(f"Generated data shape: {training_data.shape}")

        # 分类模型
        print("分类模型训练")
        try:
            x_y = pd.table_classification_model()
            X_train, X_test, y_train, y_test = x_y

            # 使用SMOTE 数据增强, 平衡is_best平衡
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

            # 验证数据分布
            logging.info(f"Classification data: y_train distribution: {np.unique(y_train, return_counts=True)}")

            # 动态批量大小
            data_size = len(X_train)
            batch_size = 32 if data_size < 1000 else 64 if data_size < 5000 else 128

            model = Model(*x_y)
            classification_history, accuracy = model.Classification_model(
                epochs=self.set["Classification_model"]["epochs"],
                batch_size=batch_size,
                data_size=data_size
            )

            # 绘制训练和验证损失曲线，检查过拟合
            plot_training_history(classification_history, "Classification", data_size=self.set["virtual"]["num_samples"])
        except Exception as e:
            logging.error(f"classification Model Error: {e}")

        # 回归模型训练
        print("\n\n回归模型训练")
        try:
            x_y = pd.table_regression_model()
            X_train, X_test, y_train, y_test = x_y

            # 验证目标值分布
            logging.info(f"Regression data: y_train mean={y_train.mean():.4f}, std={y_train.std():.4f}")

            # 动态批量大小
            data_size = len(X_train)
            batch_size = 32 if data_size < 1000 else 64 if data_size < 5000 else 128

            model = Model(*x_y)
            regression_history, r2 = model.Regression_model(
                epochs=self.set["Regression_model"]["epochs"],
                batch_size=batch_size,
                data_size=data_size
            )

            plot_training_history(regression_history, model_name="Regression", data_size=self.set["virtual"]["num_samples"], r2=r2)
        except ValueError as e:
            logging.error(f"Regression model error: {e}")
        except Exception as e:
            logging.error(f"Regression model error: {e}")


if __name__ == '__main__':
    if isfile("./config.json"):
        with open("./config.json", "r") as f:
            set = loads(f.read())

        run(set=set).main()

    else:
        logging.error("Error:Not Find config.json")





