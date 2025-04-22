#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2025/4/17 下午8:13 
# @Author : Huzhaojun
# @Version：V 1.0
# @File : data_processor.py
# @desc : README.md

"""
生成模拟训练数据，包含网络特征（如latency、packet_loss、bandwidth等）以及目标变量is_best（分类）和path_cost（回归）。
is_best基于固定阈值（0.5 ± 0.05）生成，存在类不平衡问题。
"""

import pandas as pd
import numpy as np
import random
from sklearn.ensemble import IsolationForest

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.visualization.plotter import Plotter


class DataProcessor:
    """处理数据模拟和特征工程"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger(config)
        self.plotter = Plotter(config)
        self.bandwidth_options = {
            "GigabitEthernet": 1000,
            "TenGigabitEthernet": 10000,
            "FastEthernet": 100
        }

    def detect_outliers_iqr(self, df, column, multiplier=2.0):
        """IQR（四分位距法）,IQR方法适用于单变量离群检测，适合latency、packet_loss和bandwidth等特征。"""

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        self.logger.info(f"Outliers in {column}: {len(outliers)} samples")
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)], outliers

    def detect_outliers_isolation_forest(self, df, features, contamination=0.05):
        """孤立森林适合多变量离群检测，可以同时考虑多个特征（如latency、packet_loss、bandwidth）之间的关系。"""

        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        X = df[features].values
        outlier_labels = iso_forest.fit_predict(X)
        outliers = df[outlier_labels == -1]
        self.logger.info(f"Outliers detected by Isolation Forest: {len(outliers)} samples")
        return df[outlier_labels != -1], outliers

    def handle_missing_values(self, df):
        """
        基于特征分布的填补
        对于数值特征（如latency、packet_loss），可以用中位数或均值填补。
        对于分类特征（如ospf_state），可以用众数填补。
        对于OSPF/BGP数据不完整（如as_path_length缺失），可以根据接口类型推断。
        """

        # 数值特征填补
        for col in ['latency', 'packet_loss', 'bandwidth', 'rtt', 'jitter']:
            if df[col].isna().sum() > 0:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                self.logger.info(f"Filled missing values in {col} with median: {median_value}")

        # 分类特征填补
        for col in ['ospf_state', 'is_FastEthernet', 'is_GigabitEthernet', 'is_TenGigabitEthernet']:
            if df[col].isna().sum() > 0:
                mode_value = df[col].mode()[0]
                df[col].fillna(mode_value, inplace=True)
                self.logger.info(f"Filled missing values in {col} with mode: {mode_value}")

        # 基于接口类型推断 as_path_length
        if df['as_path_length'].isna().sum() > 0:
            for idx, row in df[df['as_path_length'].isna()].iterrows():
                if row['is_TenGigabitEthernet'] == 1:
                    df.at[idx, 'as_path_length'] = df[df['is_TenGigabitEthernet'] == 1]['as_path_length'].mean()
                elif row['is_GigabitEthernet'] == 1:
                    df.at[idx, 'as_path_length'] = df[df['is_GigabitEthernet'] == 1]['as_path_length'].mean()
                else:
                    df.at[idx, 'as_path_length'] = df[df['is_FastEthernet'] == 1]['as_path_length'].mean()
            self.logger.info("Filled missing values in as_path_length based on interface type")

        return df



    def simulate_training_data(self, ospf_data, bgp_data):
        """模拟训练数据集"""

        # 获取配置和设置默认值
        num_samples = self.config.get_nested('data_collection', 'virtual', 'num_samples', default=1000)
        congestion_lambda = self.config.get_nested('data_collection', 'virtual', 'congestion_lambda', default=2)
        packet_loss_scale = self.config.get_nested('data_collection', 'virtual', 'packet_loss_scale', default=5)
        edge_case_prob = self.config.get_nested('data_collection', 'virtual', 'edge_case_prob', default=0.02)

        # 获取用户设置的条数，没有则默认1000
        num_samples = self.config.get_nested('data_collection', 'virtual', 'num_samples', default=1000)

        # 指标集
        training_data = {
            "as_path_length": [],
            "local_pref": [],
            "ospf_state": [],
            "bandwidth": [],
            "latency": [],
            "packet_loss": [],
            "jitter": [],                   # 抖动，数据包延迟的波动，单位毫秒ms
            "rtt": [],                      # 往返时间， 单位毫秒
            "cpu_usage": [],                # cpu负载率，单位%
            "memory_usage": [],             # 路由器内存使用率，单位%
            "queue_length": [],             # 队列长度，路由器输出队列中的数据包数量，反映拥塞程度
            "bandwidth_utilization": [],
            "route_stability": [],          # 路由稳定性
            "latency_bandwidth_ratio": [],  # 延迟/带宽比率
            "is_FastEthernet": [],          # 增加接口类型的单热编码列，下同
            "is_GigabitEthernet": [],
            "is_TenGigabitEthernet": [],
            "is_best": []
        }

        # 定义真实分布
        bandwidth_options = {
            "GigabitEthernet": 1000,  # 1Gbps
            "TenGigabitEthernet": 10000,  # 10Gbps
            "FastEthernet": 100  # 100Mbps
        }

        # 局部预分布常见数值
        local_pref_distribution = [80, 90, 100, 120, 150, 250]
        # 1-5跳
        as_path_length_distribution = [1, 2, 3, 4, 5]

        # 增加非FULL状态比例，模拟更多故障场景
        # FULL:70%, 非FULL:30%
        ospf_state_probs = [0.7, 0.3]

        # 模拟网络拓扑和动态
        interfaces = list(bandwidth_options.keys())
        interface = random.choices(interfaces, weights=[0.33, 0.33, 0.34])[0]  # 平均分配概率
        
        # 仿真逻辑
        for _ in range(num_samples):
            # AS路径长度（基于真实分布）
            if bgp_data:
                # 随机模拟路由
                bgp_route = random.choice(bgp_data)
                as_path_length = len(bgp_route["as_path"].split())
            else:
                as_path_length = random.choices(as_path_length_distribution, weights=[0.1, 0.4, 0.3, 0.15, 0.05])[0]
            training_data["as_path_length"].append(as_path_length)

            # 本地优先级（基于分布）
            if bgp_data and "local_pref" in bgp_data:
                local_pref = bgp_data["local_pref"]
            else:
                local_pref = random.choices(local_pref_distribution, weights=[0.1, 0.2, 0.4, 0.15, 0.1, 0.05])[0]
            training_data["local_pref"].append(local_pref)

            # OSPF状态（模拟拓扑动态）
            if ospf_data:
                ospf_neighbor = random.choice(ospf_data)
                ospf_state = 1 if "FULL" in ospf_neighbor["state"] else 0
            else:
                ospf_state = random.choices([1, 0], weights=ospf_state_probs)[0]
            training_data["ospf_state"].append(ospf_state)

            # 带宽（基于接口类型）
            if ospf_data and "interface" in ospf_data:
                interface = ospf_data["interface"]
                for iface_type in bandwidth_options:
                    if interface.startswith(iface_type):
                        bandwidth = bandwidth_options[iface_type]
                        break
                else:
                    bandwidth = bandwidth_options["FastEthernet"]  # 默认
                    interface = "FastEthernet"
            else:
                interface = random.choice(interfaces)
                bandwidth = bandwidth_options[interface]

            # 单热编码
            training_data["is_FastEthernet"].append(1 if interface == "FastEthernet" else 0)
            training_data["is_GigabitEthernet"].append(1 if interface == "GigabitEthernet" else 0)
            training_data["is_TenGigabitEthernet"].append(1 if interface == "TenGigabitEthernet" else 0)

            # 模拟拥塞
            bandwidth = bandwidth_options[interface]
            # 基准延迟
            latency = 10
            # 非拥塞场景添加微小丢包
            packet_loss =np.random.exponential(scale=0.5) if random.random() < 0.5 else 0
            # 拥塞概率
            congestion_prob = 0.3

            # 抖动, 基准抖动1-5ms，拥塞时候FastEthernet:10-30,GigabitEthernet: 5-20ms,TenGigabitEtherent: 2-10ms
            # 链路故障20-50ms
            # 基于接口和拥塞状态随机生成, 基准抖动
            jitter = random.uniform(1, 5)

            # RTT 基准RTT 20-50ms，拥塞时候增加20-100ms，链路故障时增加100-200ms
            rtt = 2 * latency + random.uniform(10, 30)

            # cpu_usage cpu利用率10-50%,拥塞增加20-40%，链路故障降低5-20%
            cpu_usage = random.uniform(10, 50)  # 基准 CPU 利用率

            # 内存利用率，20-60，拥塞时候增加10-30%，链路故障时候增加或略减10-40%
            memory_usage = random.uniform(20, 60)  # 基准内存利用率

            # 队列长度，0-10个，拥塞时增加到20-100个，Fast更高，Ten较低，链路故障时长度可能为空0-5
            queue_length = random.uniform(0, 10)  # 基准队列长度

            # 路径稳定性
            # 正常情况：振荡频率较低（0 - 2次 / 小时），模拟稳定路由
            # 拥塞情况：振荡频率略增（2 - 5次 / 小时），因为拥塞可能触发路由表更新
            # 故障情况：振荡频率显著增加（5 - 10次 / 小时），模拟链路不稳定
            route_stability = random.uniform(0, 2)

            bandwidth_utilization = random.uniform(10, 90)  # 10%-90%

            # 拥塞模拟 (泊松分布)
            if random.random() < congestion_prob:
                # 从泊松分布中生成随机样本，lam: 期望事件发生的次数（λ），必须大于等于 0。可以是一个浮点数或浮点数数组。
                # out: 从泊松分布中生成的样本，类型为 ndarray 或标量。
                congestion_intensity = np.random.poisson(lam=congestion_lambda)
                cpu_usage += random.uniform(20, 40)
                memory_usage += random.uniform(10, 30)

                if interface == "FastEthernet":
                    bandwidth_reduction = min(0.2 + 0.1 * congestion_intensity, 0.8)
                    latency_increase = 10 + 5 * congestion_intensity                # 延迟增加
                    packet_loss = np.random.exponential(scale=packet_loss_scale)    # 丢包率
                    jitter += 5 + 2 * congestion_intensity
                    rtt += 20 + 10 * congestion_intensity
                    queue_length += 50 + 10 * congestion_intensity
                    route_stability += 2 + congestion_intensity

                elif interface == "GigabitEthernet":
                    bandwidth_reduction = min(0.1 + 0.05 * congestion_intensity, 0.5)
                    latency_increase = 5 + 3 * congestion_intensity
                    packet_loss = np.random.exponential(scale=packet_loss_scale / 2.5)
                    jitter += 3 + 1.5 * congestion_intensity
                    rtt += 10 + 5 * congestion_intensity
                    queue_length += 30 + 7 * congestion_intensity
                    route_stability += 1 + 0.5 * congestion_intensity

                else:  # TenGigabitEthernet
                    bandwidth_reduction = min(0.05 + 0.03 * congestion_intensity, 0.3)
                    latency_increase = 2 + 2 * congestion_intensity
                    packet_loss = np.random.exponential(scale=packet_loss_scale / 10)
                    jitter += 1 + congestion_intensity
                    rtt += 5 + 3 * congestion_intensity
                    queue_length += 20 + 5 * congestion_intensity
                    route_stability += 0.5 + 0.3 * congestion_intensity

                bandwidth = bandwidth * (1 - bandwidth_reduction)
                latency += latency_increase
                # packet_loss = packet_loss

            # 边缘情况增强
            if random.random() < edge_case_prob:

                # 间歇性丢包
                if random.random() < 0.3:
                    packet_loss += np.random.exponential(scale=5)  # 小幅度高频丢包
                    self.logger.debug(f"Edge case applied: intermittent packet loss, packet_loss={packet_loss:.2f}")
                else:
                    # 灾难性丢包
                    packet_loss += random.uniform(30, 50)  # 高幅度丢包
                    self.logger.debug(f"Edge case applied: catastrophic packet loss, packet_loss={packet_loss:.2f}")

                bandwidth_utilization = min(bandwidth_utilization + random.uniform(50, 80), 100)
                queue_length += random.uniform(100, 200)
                latency += random.uniform(20, 50)
                packet_loss += random.uniform(10, 30)  # 加大扰动幅度
                jitter += random.uniform(10, 30)
                rtt += random.uniform(50, 100)
                route_stability += random.uniform(5, 10)
                self.logger.debug(f"Edge case applied: traffic spike, utilization={bandwidth_utilization:.2f}")

            # 15%概率模拟链路故障
            if random.random() < 0.15:
                ospf_state = 0
                bandwidth = bandwidth * 0.1                       # 带宽降至10%
                latency += random.uniform(50, 100)          # 延迟显著增加
                packet_loss += random.uniform(10, 25)       # 丢包率增加
                jitter += random.uniform(20, 50)            # 抖动增加
                rtt += random.uniform(100, 200)             # 往返时间增加
                route_stability += random.uniform(10, 15)   # 故障时振荡显著增加
                self.logger.debug(f"Edge case applied: multi-link failure, latency={latency:.2f}")

            # 添加平滑噪声，确保分布更连续
            packet_loss += np.random.normal(loc=0, scale=3.0)       # 添加正态分布噪声
            packet_loss = min(packet_loss, 50)                      # 设置丢包率上限,现实中很少超过50%

            # 分段噪声
            if packet_loss < 10:
                packet_loss += np.random.normal(loc=0, scale=0.5)  # 低丢包率区域小噪声
            else:
                packet_loss += np.random.normal(loc=0, scale=2.0)  # 高丢包率区域大噪声
            packet_loss = max(0, packet_loss)                       # 确保0 <= x <= 50

            training_data["bandwidth"].append(round(bandwidth))
            training_data["latency"].append(round(latency, 2))
            training_data["packet_loss"].append(round(packet_loss, 2))
            training_data["jitter"].append(round(jitter, 2))
            training_data["rtt"].append(round(rtt, 2))
            training_data["cpu_usage"].append(round(cpu_usage, 2))
            training_data["memory_usage"].append(round(memory_usage, 2))
            training_data["queue_length"].append(round(queue_length, 2))
            training_data["route_stability"].append(round(route_stability, 2))
            training_data["bandwidth_utilization"].append(round(bandwidth_utilization, 2))

            # 延迟/带宽比率
            # 计算公式：latency / bandwidth，单位为ms / Mbps
            # 调整：由于bandwidth范围较大（100 - 10000Mbps）.直接比率可能分布不均.对比率取对数（log1p(latency / bandwidth)）作为衍生特征
            latency_bandwidth_ratio = latency / bandwidth if bandwidth > 0 else 0  # 避免除零
            training_data["latency_bandwidth_ratio"].append(round(latency_bandwidth_ratio, 4))

            # 动态阈值，基于样本分布
            # 调整is_best评分，考虑拥塞影响
            score = (
                    0.26 * (local_pref / 200) +
                    0.16 * (1 - as_path_length / 5) +
                    0.10 * ospf_state +
                    0.10 * (bandwidth / bandwidth_options["TenGigabitEthernet"]) -
                    0.05 * (latency / 100) -
                    0.05 * (packet_loss / 50) -
                    0.03 * (bandwidth_utilization / 100) -
                    0.03 * (jitter / 50) -
                    0.03 * (rtt / 200) -
                    0.02 * (cpu_usage / 100) -
                    0.02 * (memory_usage / 100) -
                    0.02 * (queue_length / 100) -
                    0.03 * (route_stability / 10) -         # 路径稳定性, 负向特征
                    0.03 * (latency_bandwidth_ratio / 1) +    # 延迟/带宽比率，负向特征
                    0.02 * training_data["is_GigabitEthernet"][-1] +  # 正向贡献
                    0.03 * training_data["is_TenGigabitEthernet"][-1]  # 更高带宽接口贡献更大
            )

            threshold = 0.5 - 0.1 * (bandwidth_utilization / 100)  # 高拥塞降低阈值
            is_best = 1 if score > threshold else 0                 # 动态阈值
            training_data["is_best"].append(is_best)

        # 转换为DataFrame
        df = pd.DataFrame(training_data)
        self.logger.info(f"Initial DataFrame shape: {df.shape}")

        # 验证 数据包丢失率分布
        self.plotter.plot_edge_cases(df)
        self.plotter.plot_edge_boxplot(df)

        # 检查 NaN 值
        if df.isna().any().any():
            self.logger.warning(f"NaN values detected:\n{df.isna().sum()}")
            df = df.dropna()
            self.logger.info(f"After dropping NaN: DataFrame shape: {df.shape}")

        # 数据验证
        assert len(df) == num_samples, "Length mismatch"
        assert df["as_path_length"].min() >= 1, "Invalid as_path_length"
        assert df["local_pref"].min() >= 0, "Invalid local_pref"
        assert set(df["ospf_state"]).issubset({0, 1}), "Invalid ospf_state"
        assert df["bandwidth"].min() >= 1, "Invalid bandwidth"
        assert set(df["is_best"]).issubset({0, 1}), "Invalid is_best"
        assert df["latency"].min() >= 0, "Invalid latency"
        assert df["packet_loss"].min() >= 0, "Invalid packet_loss"
        assert df["jitter"].min() >= 0, "Invalid jitter"
        assert df["rtt"].min() >= 0, "Invalid rtt"
        assert df["cpu_usage"].min() >= 0 and df["cpu_usage"].max() <= 100, "Invalid cpu_usage"
        assert df["memory_usage"].min() >= 0 and df["memory_usage"].max() <= 100, "Invalid memory_usage"
        assert df["queue_length"].min() >= 0, "Invalid queue_length"
        assert df["route_stability"].min() >= 0, "Invalid route_stability"
        assert df["latency_bandwidth_ratio"].min() >= 0, "Invalid latency_bandwidth_ratio"
        # 单热编码列无需额外变换，可直接用于模型训练
        assert set(df["is_FastEthernet"]).issubset({0, 1}), "Invalid is_FastEthernet"
        assert set(df["is_GigabitEthernet"]).issubset({0, 1}), "Invalid is_GigabitEthernet"
        assert set(df["is_TenGigabitEthernet"]).issubset({0, 1}), "Invalid is_TenGigabitEthernet"

        # 添加衍生特征
        df["bandwidth_log"] = np.log1p(df["bandwidth"])
        df["jitter_log"] = np.log1p(df["jitter"])
        df["rtt_log"] = np.log1p(df["rtt"])
        df["resource_usage"] = 0.6 * df["cpu_usage"] + 0.4 * df["memory_usage"]
        df["route_stability_log"] = np.log1p(df["route_stability"])  # 新增
        df["latency_bandwidth_ratio_log"] = np.log1p(df["latency_bandwidth_ratio"])  # 新增
        df["latency_packet_loss"] = df["latency"] * df["packet_loss"]
        # 添加对 packet_loss 的对数变换
        df["packet_loss_log"] = np.log1p(df["packet_loss"])
        self.logger.info(
            f"Added packet_loss_log: mean={df['packet_loss_log'].mean():.4f}, std={df['packet_loss_log'].std():.4f}")
        df["utilization_queue"] = df["bandwidth_utilization"] * df["queue_length"]


        # 确保 bandwidth 不为 0
        if (df["bandwidth"] <= 0).any():
            self.logger.warning(f"Invalid bandwidth values detected: {df['bandwidth'][df['bandwidth'] <= 0]}")
            df["bandwidth"] = df["bandwidth"].clip(lower=1)  # 将 bandwidth 限制为最小值 1

        # 如果bandwidth 很接近与0，可能导致np.log1p(1000 / df["bandwidth"]) NaN或者无穷大进而引发path_cost异常甚至错误

        # 生成 path_cost
        df["path_cost"] = (
                0.5 * df["as_path_length"] +
                np.log1p(1000 / df["bandwidth"]) +
                (1 - df["ospf_state"]) * 0.2 +
                0.1 * df["latency"] +
                0.1 * df["packet_loss"] +
                0.05 * df["bandwidth_utilization"] +
                0.05 * df["jitter"] +
                0.05 * df["rtt"] +
                0.03 * df["resource_usage"] +
                0.02 * df["queue_length"] +
                0.03 * df["route_stability"] +
                0.03 * df["latency_bandwidth_ratio"] +
                0.02 * df["is_GigabitEthernet"] +
                0.03 * df["is_TenGigabitEthernet"]

        )

        self.logger.info("path_cost column created successfully")
        self.logger.info(f"path_cost head: {df['path_cost'].head()}")

        # 先填补不完整数据
        df = self.handle_missing_values(df)

        # 结合两种方法，先用IQR过滤明显的单变量异常值(如packet_loss超过50的样本),再用孤立森林检测多变量异常（如latency和packet_loss同时异常）


        # df = self.detect_outliers_iqr(df, 'packet_loss')
        # df = self.detect_outliers_iqr(df, 'latency')
        # df = self.detect_outliers_iqr(df, 'bandwidth')

        # IQR
        df, outliers_iqr = self.detect_outliers_iqr(df, 'packet_loss', multiplier=2.0)
        # 孤立森林
        df, outliers_iso = self.detect_outliers_isolation_forest(df, ['latency', 'packet_loss', 'bandwidth'],
                                                                 contamination=0.05)

        outliers_combined = pd.concat([outliers_iqr, outliers_iso]).drop_duplicates()
        self.logger.info(f"Outliers summary:\n{outliers_combined.describe()}")

        # features_to_check = ['latency', 'packet_loss', 'bandwidth']
        # df = self.detect_outliers_isolation_forest(df, features_to_check)

        # 检查 path_cost 是否包含 NaN
        if df["path_cost"].isna().any():
            self.logger.warning(
                f"NaN values in path_cost: {df[df['path_cost'].isna()][['bandwidth', 'latency', 'route_stability']]}")
            df = df.dropna()
            self.logger.info(f"After dropping NaN in path_cost: DataFrame shape: {df.shape}")

        corr_is_best = df.corr()['is_best'].drop(['is_best', 'path_cost']).abs().sort_values(ascending=False)
        corr_path_cost = df.corr()['path_cost'].drop(['is_best', 'path_cost']).abs().sort_values(ascending=False)
        self.logger.info(f"Feature correlation with is_best:\n{corr_is_best}")
        self.logger.info(f"Feature correlation with path_cost:\n{corr_path_cost}")

        # 日志打印鉴别
        for interface in ['FastEthernet', 'GigabitEthernet', 'TenGigabitEthernet']:
            mask = df[f'is_{interface}'] == 1
            self.logger.info(
                f"{interface} packet_loss: mean={df[mask]['packet_loss'].mean():.4f}, std={df[mask]['packet_loss'].std():.4f}")

        self.logger.info(
            f"packet_loss statistics: mean={df['packet_loss'].mean():.4f}, std={df['packet_loss'].std():.4f}")
        self.logger.info(f"packet_loss quantiles: {df['packet_loss'].quantile([0.25, 0.5, 0.75, 0.95])}")
        edge_cases = df[df["packet_loss"] > 10]
        self.logger.info(f"High packet loss (>10): {len(edge_cases)} samples, {len(edge_cases) / len(df) * 100:.2f}%")

        self.logger.info(f"Generated data summary:\n{df.describe()}")
        self.logger.info(f"is_best distribution: {df['is_best'].value_counts(normalize=True)}")
        self.logger.info(f"jitter summary: mean={df['jitter'].mean():.4f}, std={df['jitter'].std():.4f}")
        self.logger.info(f"rtt summary: mean={df['rtt'].mean():.4f}, std={df['rtt'].std():.4f}")
        self.logger.info(f"cpu_usage summary: mean={df['cpu_usage'].mean():.4f}, std={df['cpu_usage'].std():.4f}")
        self.logger.info(
            f"memory_usage summary: mean={df['memory_usage'].mean():.4f}, std={df['memory_usage'].std():.4f}")
        self.logger.info(
            f"queue_length summary: mean={df['queue_length'].mean():.4f}, std={df['queue_length'].std():.4f}")
        self.logger.info(
            f"route_stability summary: mean={df['route_stability'].mean():.4f}, std={df['route_stability'].std():.4f}")
        self.logger.info(
            f"latency_bandwidth_ratio summary: mean={df['latency_bandwidth_ratio'].mean():.4f}, std={df['latency_bandwidth_ratio'].std():.4f}")
        self.logger.info(f"path_cost summary: mean={df['path_cost'].mean():.4f}, std={df['path_cost'].std():.4f}")

        self.logger.info(f"Final DataFrame shape: {df.shape}")
        if len(df) != num_samples:
            self.logger.warning(f"Expected {num_samples} samples, got {len(df)}")

        return df
