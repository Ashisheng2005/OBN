#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2025/4/17 下午8:14 
# @Author : Huzhaojun
# @Version：V 1.0
# @File : trainer.py
# @desc : README.md

"""
协调分类和回归模型的训练，使用SMOTE处理is_best的类不平衡，标准化数据并调用模型训练。
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks
from xgboost import XGBClassifier, XGBRegressor
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.visualization.plotter import Plotter
from src.preprocessing.scaler import DataScaler
from src.modeling.regression import RegressionModel
from src.modeling.classification import ClassificationModel
import optuna
from tqdm import tqdm


class Trainer:
    """协调模型训练和数据预处理。"""

    def __init__(self, config: Config, data):
        self.config = config
        self.logger = setup_logger(config)
        self.data = data
        self.plotter = Plotter(config)
        # 确认是否启用GPU
        self.logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

    # def apply_pca(self, X, n_components=0.95):
    #     """使用PCA进一步降维"""
    #
    #     pca = PCA(n_components=n_components)
    #     X_pca = pca.fit_transform(X)
    #     self.logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    #     return X_pca

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, task='classification', class_weights=None):
        # def objective(trial):
        #     if task == 'classification':
        #         model = ClassificationModel(self.config, X_train.shape, trial)
        #         history, accuracy = model.train(X_train, y_train, X_val, y_val, X_val, y_val,
        #                                         class_weights={0: 1.0, 1: 1.0})
        #         return accuracy
        #     else:
        #         model = RegressionModel(self.config, X_train.shape, trial)
        #         history, r2 = model.train(X_train, y_train, X_val, y_val, X_val, y_val)
        #         return r2
        #
        # study = optuna.create_study(direction='maximize')
        # study.optimize(objective, n_trials=50)
        # self.logger.info(f"Best hyperparameters: {study.best_params}")
        # return study.best_params
        def objective(trial):
            self.logger.info(f"Starting Optuna trial {trial.number}")
            if task == 'classification':
                model = ClassificationModel(self.config, X_train.shape, trial)
                history, accuracy = model.train(X_train, y_train, X_val, y_val, X_val, y_val, class_weights)
                return accuracy
            else:
                model = RegressionModel(self.config, X_train.shape, trial)
                history, r2 = model.train(X_train, y_train, X_val, y_val, X_val, y_val)
                return r2

        study = optuna.create_study(direction='maximize')
        # study.optimize(objective, n_trials=10)
        for _ in tqdm(range(5), desc="Optuna Trials"):  # 降低试验次数
            study.optimize(objective, n_trials=1)
        self.logger.info(f"Best hyperparameters: {study.best_params}")
        return study.best_params

    # def select_features_mutual_info(self, X, y, task='classification'):
    #     """
    #         使用互信息评分降低维数,目前的特征选择基于相关性阈值（corr > 0.1），但这可能忽略非线性关系。
    #     使用互信息评分（Mutual Information）来识别冗余特征，并结合主成分分析（PCA）降低维数。
    #     """
    #
    #     # 互信息评分
    #     if task == 'classification':
    #         mi = mutual_info_classif(X, y, random_state=42)
    #     else:
    #         mi = mutual_info_regression(X, y, random_state=42)
    #     mi_scores = pd.Series(mi, index=X.columns)
    #
    #     # 根据任务类型选择随机森林模型
    #     if task == 'classification':
    #         rf = RandomForestClassifier(random_state=42)
    #     else:
    #         rf = RandomForestRegressor(random_state=42)
    #
    #     rf.fit(X, y)
    #     rf_importances = pd.Series(rf.feature_importances_, index=X.columns)
    #
    #     # 综合评分
    #     combined_scores = 0.5 * mi_scores + 0.5 * rf_importances
    #     self.logger.info(f"Combined feature scores:\n{combined_scores.sort_values(ascending=False)}")
    #
    #     # 选择综合得分较高的特征
    #     selected_features = combined_scores[combined_scores > 0.05].index.tolist()
    #     self.logger.info(f"Selected features: {selected_features}")
    #     return selected_features

    # def prepare_classification_data(self):
    #     """分类模型使用的数据标签,分类模型（输出 sigmoid）无法处理非0/1的标签"""
    #
    #     # 计算相关性并筛选特征
    #     # 使用互信息评分
    #     corr = self.data.corr()['is_best'].drop(['is_best', 'path_cost']).abs()
    #     mi_rf_selected_features = self.select_features_mutual_info(self.data.drop(['is_best', 'path_cost'], axis=1),
    #                                                             self.data['is_best'], task='classification')
    #     selected_features = list(set(corr[corr > 0.1].index.tolist()) | set(mi_rf_selected_features))  # 取并集
    #
    #     self.logger.info(f"Selected features for classification: {selected_features}")
    #     X = self.data[selected_features].values
    #     y = self.data["is_best"].values
    #
    #     # 三向拆分：60%训练，20%验证，20%测试
    #     X_temp, X_test, y_temp, y_test = train_test_split(
    #         X, y, test_size=0.2, stratify=y, random_state=42
    #     )
    #     X_train, X_val, y_train, y_val = train_test_split(
    #         X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    #     )
    #
    #     # 数据标准化
    #     scaler = DataScaler(self.config)
    #     X_train, y_train = scaler.fit_transform(X_train, y_train)
    #     X_val, y_val = scaler.transform(X_val, y_val)
    #     X_test, y_test = scaler.transform(X_test, y_test)
    #
    #     # 使用ADASYN代替Borderline-SMOTE
    #     sampler = ADASYN(random_state=42, sampling_strategy=0.8)
    #     X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
    #
    #     # 再使用 TomekLinks 欠采样
    #     tomek = TomekLinks()
    #     X_train_final, y_train_final = tomek.fit_resample(X_train_resampled, y_train_resampled)
    #
    #     # 调整类权重
    #     pos_weight = (len(y_train_final) / np.sum(y_train_final == 1)) * 0.7  # 增加正样本权重
    #     class_weights = {0: 1.0, 1: pos_weight}
    #     self.logger.info(f"Updated class weights: {class_weights}")
    #
    #     # 验证合成样本分布
    #     # feature_names = self.data.drop(["is_best", "path_cost"], axis=1).columns
    #     for i, col in enumerate(selected_features):
    #         stat, p = ks_2samp(X_train_resampled[:, i], X_train_final[:, i])
    #         self.logger.info(f"KS test for {col}: stat={stat:.4f}, p={p:.4f}")
    #
    #     scaler.save('classification')
    #     self.logger.info(
    #         f"Classification data: y_train distribution: {np.unique(y_train_final, return_counts=True)}")
    #     # return X_train_final, X_val, X_test, y_train_final, y_val, y_test
    #
    #     return X_train_final, X_val, X_test, y_train_final, y_val, y_test, class_weights

    def prepare_classification_data(self):
        corr = self.data.corr()['is_best'].drop(['is_best', 'path_cost']).abs()
        selected_features = corr[corr > 0.1].index.tolist()
        X = self.data[selected_features].values
        y = self.data["is_best"].values

        # 检查并处理 NaN 值
        if np.isnan(X).any():
            self.logger.warning("NaN values detected in X before ADASYN. Applying imputation.")
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)

        # K折交叉验证
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        X_train_final, y_train_final = [], []
        for train_idx, _ in kfold.split(X, y):
            X_temp, y_temp = X[train_idx], y[train_idx]
            scaler = DataScaler(self.config)
            X_temp, y_temp = scaler.fit_transform(X_temp, y_temp)
            sampler = ADASYN(random_state=42, sampling_strategy=0.5)    # 提高少数类比例
            X_resampled, y_resampled = sampler.fit_resample(X_temp, y_temp)
            tomek = TomekLinks()
            X_final, y_final = tomek.fit_resample(X_resampled, y_resampled)
            X_train_final.append(X_final)
            y_train_final.append(y_final)

        X_train = np.vstack(X_train_final)
        y_train = np.hstack(y_train_final)

        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp,
                                                          random_state=42)

        scaler = DataScaler(self.config)
        X_train, y_train = scaler.fit_transform(X_train, y_train)
        X_val, y_val = scaler.transform(X_val, y_val)
        X_test, y_test = scaler.transform(X_test, y_test)

        pos_weight = (len(y_train) / np.sum(y_train == 1)) * 0.7
        class_weights = {0: 1.0, 1: pos_weight}
        scaler.save('classification')
        return X_train, X_val, X_test, y_train, y_val, y_test, class_weights

    # def prepare_regression_data(self):
    #     """回归模型数据，使用path_cost作为目标"""
    #
    #     if "path_cost" not in self.data.columns:
    #         raise ValueError("Regression requires 'path_cost' column.")
    #
    #     # 确保数据没有 NaN 值
    #     data_clean = self.data.dropna(subset=["is_best", "path_cost"])
    #     self.logger.info(f"After dropping NaN: Data shape={data_clean.shape}")
    #
    #     # 使用互信息评分
    #     corr = data_clean.corr()['path_cost'].drop(['is_best', 'path_cost']).abs()
    #     mi_selected_features = self.select_features_mutual_info(data_clean.drop(['is_best', 'path_cost'], axis=1),
    #                                                             data_clean['path_cost'], task='regression')
    #     selected_features = list(set(corr[corr > 0.1].index.tolist()) & set(mi_selected_features))
    #
    #     self.logger.info(f"Selected features for regression: {selected_features}")
    #     X = data_clean[selected_features].values
    #     y = data_clean["path_cost"].values
    #     weights = np.where(data_clean["is_best"] == 1, 1.5, 1.0)
    #
    #     X_temp, X_test, y_temp, y_test, weights_temp, weights_test = train_test_split(
    #         X, y, weights, test_size=0.2, random_state=42
    #     )
    #     X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
    #         X_temp, y_temp, weights_temp, test_size=0.25, random_state=42
    #     )
    #
    #     # 检查是否存在 NaN 值
    #     if np.any(np.isnan(X)) or np.any(np.isnan(y)):
    #         self.logger.warning(f"NaN values detected: X={np.isnan(X).sum()}, y={np.isnan(y).sum()}")
    #
    #     # 确保 X 和 y 样本数一致
    #     if len(X) != len(y):
    #         self.logger.error(f"X and y mismatch: X={len(X)}, y={len(y)}")
    #         valid_indices = min(len(X), len(y))
    #         X = X[:valid_indices]
    #         y = y[:valid_indices]
    #         self.logger.info(f"Truncated to match: X shape={X.shape}, y shape={y.shape}")
    #
    #     # X_train, X_test, y_train, y_test = train_test_split(
    #     #     X, y, test_size=0.2, random_state=42
    #     # )
    #
    #     # self.logger.info(f"After split: X_train shape={X_train.shape}, y_train shape={y_train.shape}, "
    #     #                  f"X_test shape={X_test.shape}, y_test shape={y_test.shape}")
    #
    #     scaler = DataScaler(self.config)
    #     X_train, y_train = scaler.fit_transform(X_train, y_train, is_regression=True)
    #     X_val, y_val = scaler.transform(X_val, y_val, is_regression=True)
    #     X_test, y_test = scaler.transform(X_test, y_test, is_regression=True)
    #
    #     scaler.save('regression')
    #     self.logger.info(f"Regression data: y_train mean={y_train.mean():.4f}, std={y_train.std():.4f}")
    #     return X_train, X_val, X_test, y_train, y_val, y_test, weights_train, weights_val, weights_test

    def prepare_regression_data(self):
        data_clean = self.data.dropna(subset=["is_best", "path_cost"])
        corr = data_clean.corr()['path_cost'].drop(['is_best', 'path_cost']).abs()
        selected_features = corr[corr > 0.1].index.tolist()
        X = data_clean[selected_features].values
        y = data_clean["path_cost"].values
        weights = np.where(data_clean["is_best"] == 1, 1.5, 1.0)

        X_temp, X_test, y_temp, y_test, weights_temp, weights_test = train_test_split(
            X, y, weights, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
            X_temp, y_temp, weights_temp, test_size=0.25, random_state=42
        )

        scaler = DataScaler(self.config)
        X_train, y_train = scaler.fit_transform(X_train, y_train, is_regression=True)
        X_val, y_val = scaler.transform(X_val, y_val, is_regression=True)
        X_test, y_test = scaler.transform(X_test, y_test, is_regression=True)

        scaler.save('regression')
        return X_train, X_val, X_test, y_train, y_val, y_test, weights_train, weights_val, weights_test

    def train_xgboost(self, X_train, y_train, X_val, y_val, task='classification'):
        if task == 'classification':
            model = XGBClassifier(random_state=42)
            model.fit(X_train, y_train)
            accuracy = model.score(X_val, y_val)
            self.logger.info(f"XGBoost Classification Accuracy: {accuracy:.4f}")
            return model, accuracy
        else:
            model = XGBRegressor(random_state=42)
            model.fit(X_train, y_train)
            r2 = model.score(X_val, y_val)
            self.logger.info(f"XGBoost Regression R2: {r2:.4f}")
            return model, r2

    # def train(self):
    #     # Classification
    #     try:
    #         self.logger.info("Training classification model")
    #         X_train, X_val, X_test, y_train, y_val, y_test, class_weights = self.prepare_classification_data()
    #         model = ClassificationModel(self.config, X_train.shape)
    #         history, accuracy = model.train(X_train, y_train, X_val, y_val, X_test, y_test, class_weights)
    #         model.save_model()
    #
    #         # 绘结果图
    #         self.plotter.plot_training_history(history, "Classification", len(X_train), accuracy=accuracy)
    #
    #         # 通过绘制PR和ROC曲线评估分类模型在is_best=1上的性能
    #         y_test_pred_proba = model.model.predict(X_test)
    #         self.plotter.plot_pr_roc(y_test, y_test_pred_proba, "Classification")
    #
    #     except Exception as e:
    #         self.logger.error(f"Classification training error: {e}")
    #         raise
    #
    #     # Regression
    #     try:
    #         self.logger.info("Training regression model")
    #         X_train, X_val, X_test, y_train, y_val, y_test, weights_train, weights_val, weights_test = self.prepare_regression_data()
    #
    #         # 获取测试集的is_best标签
    #         data_clean = self.data.dropna(subset=["is_best", "path_cost"])
    #         _, X_test_indices = train_test_split(
    #             range(len(data_clean)), test_size=0.2, random_state=42
    #         )
    #         y_test_is_best = data_clean["is_best"].values[X_test_indices]
    #
    #         model = RegressionModel(self.config, X_train.shape)
    #         history, r2 = model.train(X_train, y_train, X_val, y_val, X_test, y_test, weights_train, y_test_is_best)
    #         model.save_model()
    #         self.plotter.plot_training_history(history, "Regression", len(X_train), r2=r2)
    #     except Exception as e:
    #         self.logger.error(f"Regression training error: {e}")
    #         raise

    def train(self):
        # Classification
        self.logger.info("Training classification model")
        X_train, X_val, X_test, y_train, y_val, y_test, class_weights = self.prepare_classification_data()
        best_params = self.optimize_hyperparameters(X_train, y_train, X_val, y_val, task='classification')
        model = ClassificationModel(self.config, X_train.shape, trial=None)  # 使用最佳参数
        history, accuracy = model.train(X_train, y_train, X_val, y_val, X_test, y_test, class_weights)
        model.save_model()
        self.plotter.plot_training_history(history, "Classification", len(X_train), accuracy=accuracy)

        # XGBoost Classification
        xgb_model, xgb_accuracy = self.train_xgboost(X_train, y_train, X_val, y_val, task='classification')

        # Regression
        self.logger.info("Training regression model")
        X_train, X_val, X_test, y_train, y_val, y_test, weights_train, weights_val, weights_test = self.prepare_regression_data()
        best_params = self.optimize_hyperparameters(X_train, y_train, X_val, y_val, task='regression')
        model = RegressionModel(self.config, X_train.shape, trial=None)
        data_clean = self.data.dropna(subset=["is_best", "path_cost"])
        _, X_test_indices = train_test_split(range(len(data_clean)), test_size=0.2, random_state=42)
        y_test_is_best = data_clean["is_best"].values[X_test_indices]
        history, r2 = model.train(X_train, y_train, X_val, y_val, X_test, y_test, weights_train, y_test_is_best)
        model.save_model()
        self.plotter.plot_training_history(history, "Regression", len(X_train), r2=r2)

        # XGBoost Regression
        xgb_model, xgb_r2 = self.train_xgboost(X_train, y_train, X_val, y_val, task='regression')

    def predict_optimal_path(self, X):
        # 加载标准化器和模型

        from joblib import load
        model_dir = self.config.get_nested('output', 'model_dir', default='./model/')
        scaler = load(f"{model_dir}scaler_classification.pkl")
        X_scaled = scaler.transform(X)

        clf_model = ClassificationModel(self.config, X_scaled.shape)
        clf_model.model = tf.keras.models.load_model(f"{model_dir}best_path_classification_model.keras")
        reg_model = RegressionModel(self.config, X_scaled.shape)
        reg_model.model = tf.keras.models.load_model(f"{model_dir}best_path_regression_model.keras")

        # MC Dropout
        def mc_dropout_predict(model, X, n_samples=50):
            model.trainable = False
            predictions = []
            for _ in range(n_samples):
                predictions.append(model.predict(X, training=True))  # 启用Dropout
            return np.mean(predictions, axis=0), np.std(predictions, axis=0)

        clf_probs, clf_uncertainty = mc_dropout_predict(clf_model.model, X_scaled)
        reg_scores, reg_uncertainty = mc_dropout_predict(reg_model.model, X_scaled)

        reg_scores_normalized = (reg_scores - reg_scores.min()) / (reg_scores.max() - reg_scores.min() + 1e-10)
        final_scores = 0.6 * clf_probs + 0.4 * (1 - reg_scores_normalized)
        minority_bonus = np.where(clf_probs > 0.5, 0.1, 0.0)
        final_scores += minority_bonus

        optimal_path_idx = np.argmax(final_scores)
        self.logger.info(
            f"Optimal path index: {optimal_path_idx}, score: {final_scores[optimal_path_idx]:.4f}, uncertainty: {clf_uncertainty[optimal_path_idx]:.4f}")
        return optimal_path_idx


