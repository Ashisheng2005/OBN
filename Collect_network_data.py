#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2025/4/11 下午5:04 
# @Author : Huzhaojun
# @Version：V 1.0
# @File : Collect_network_data.py
# @desc : README.md

from netmiko import ConnectHandler
from typing import Any
import pandas as pd
import numpy as np
from multiprocessing import Pool
import re
import random
import logging

class Collect_network_data:

    def __init__(self, mode: str = "virtual", mode_set: dict = None):
        self.mode = mode        # virtual 虚拟 / real 实时
        self.mode_set = mode_set
        logging.basicConfig(level=logging.INFO)

        try:
            self.device = {
                'device_type': 'cisco_ios',
                'host': '192.168.1.1',
                'username': 'admin',
                'password': 'password',
            } if mode == "virtual" else mode_set["real"]["device"]

        except KeyError as e:
            logging.warning(f"KeyError from mode_set : {e}")

    def parse_ospf_output(self, output) -> Any:
        """ospf 数据处理"""
        lines = output.split("\n")
        routes = []
        # 跳过表头和状态代码，直到找到实际数据行
        data_start = False
        for line in lines:
            if not line.strip() or line.startswith("Status codes") or line.startswith(
                    "Origin codes") or line.startswith("BGP table") or line.startswith("Network"):
                continue
            if line.strip().startswith("*") or line.strip().startswith(">"):
                data_start = True
                parts = re.split(r'\s+', line.strip())
                # 确保行有足够的字段
                if len(parts) < 7:
                    continue
                try:
                    route = {
                        "network": parts[1],
                        "next_hop": parts[2],
                        "metric": int(parts[3]),
                        "local_pref": int(parts[4]),
                        "weight": int(parts[5]),
                        "as_path": " ".join(parts[6:-1]),
                        "origin": parts[-1],
                        "best": ">" in parts[0]
                    }
                    routes.append(route)
                except (ValueError, IndexError):
                    # 跳过无法解析的行
                    continue
        return routes

    def parse_bgp_output(self, output) -> Any:
        """bgp数据处理"""
        lines = output.split("\n")
        routes = []
        for line in lines:
            if not line.strip() or line.startswith("Status codes") or line.startswith(
                    "Origin codes") or line.startswith("BGP table") or line.startswith("Network"):
                continue
            if line.strip().startswith("*") or line.strip().startswith(">"):
                parts = re.split(r'\s+', line.strip())
                if len(parts) < 7:
                    continue
                try:
                    route = {
                        "network": parts[1],
                        "next_hop": parts[2],
                        "metric": int(parts[3]),
                        "local_pref": int(parts[4]),
                        "weight": int(parts[5]),
                        "as_path": " ".join(parts[6:-1]),
                        "origin": parts[-1],
                        "best": ">" in parts[0]
                    }
                    routes.append(route)
                except (ValueError, IndexError):
                    continue
        return routes

    def cleanse_bgp_routes(self, bgp_routes):
        data = []
        for route in bgp_routes:
            data.append({
                "network": route["network"],
                "as_path_length": len(route["as_path"].split()),
                "local_pref": route["local_pref"],
                "is_best": route["best"],
                "latency": 10.5  # 假设测量得到的延迟（ms）
            })

        df = pd.DataFrame(data)
        return df.head()

    def Simulation_training_data(self, ospf_neighbors, bgp_routes, num_samples=50):
        """模拟训练数据，遵循OSPF/BGP路由逻辑，避免随机噪声"""
        training_data = {
            "as_path_length": [],
            "local_pref": [],
            "ospf_state": [],
            "bandwidth": [],
            "latency": [],  # 新增：延迟（ms）
            "packet_loss": [],  # 新增：丢包率（%）
            "bandwidth_utilization": [],    # 新增带宽使用率 %
            "is_best": []
        }

        # 定义真实分布
        bandwidth_options = {
            "GigabitEthernet": 1000,      # 1Gbps
            "TenGigabitEthernet": 10000,  # 10Gbps
            "FastEthernet": 100           # 100Mbps
        }
        local_pref_distribution = [80, 90, 100, 120, 150, 200]  # 常见值
        as_path_length_distribution = [1, 2, 3, 4, 5]           # 1-5跳

        # 增加非FULL状态比例，模拟更多故障场景
        # FULL:60%, 非FULL:40%
        ospf_state_probs = [0.6, 0.4]

        # 模拟网络拓扑和动态
        interfaces = list(bandwidth_options.keys())
        for _ in range(num_samples):
            # AS路径长度（基于真实分布）
            if bgp_routes:
                bgp_route = random.choice(bgp_routes)
                as_path_length = len(bgp_route["as_path"].split())
            else:
                as_path_length = random.choices(as_path_length_distribution, weights=[0.1, 0.4, 0.3, 0.15, 0.05])[0]
            training_data["as_path_length"].append(as_path_length)

            # 本地优先级（基于分布）
            if bgp_routes and "local_pref" in bgp_route:
                local_pref = bgp_route["local_pref"]
            else:
                local_pref = random.choices(local_pref_distribution, weights=[0.1, 0.2, 0.5, 0.1, 0.05, 0.05])[0]

            # OSPF状态（模拟拓扑动态）
            if ospf_neighbors:
                ospf_neighbor = random.choice(ospf_neighbors)
                ospf_state = 1 if "FULL" in ospf_neighbor["state"] else 0
            else:
                ospf_state = random.choices([1, 0], weights=ospf_state_probs)[0]
            training_data["ospf_state"].append(ospf_state)

            # 带宽（基于接口类型）
            if ospf_neighbors and "interface" in ospf_neighbor:
                interface = ospf_neighbor["interface"]
                for iface_type in bandwidth_options:
                    if interface.startswith(iface_type):
                        bandwidth = bandwidth_options[iface_type]
                        break
                else:
                    bandwidth = bandwidth_options["FastEthernet"]  # 默认
            else:
                interface = random.choice(interfaces)
                bandwidth = bandwidth_options[interface]

            # 模拟带宽波动（±10%）
            bandwidth = bandwidth * random.uniform(0.9, 1.1)

            # 20%概率模拟拥塞
            if random.random() < 0.2:
                bandwidth = bandwidth * 0.5  # 带宽减半
                # training_data["bandwidth"].append(bandwidth)

            interface = random.choice(interfaces)
            bandwidth = bandwidth_options[interface]
            latency = 10  # 基准延迟
            packet_loss = 0  # 基准丢包率

            interface = random.choice(interfaces)
            bandwidth = bandwidth_options[interface]
            latency = 10  # 基准延迟
            packet_loss = 0  # 基准丢包率

            # 模拟拥塞
            congestion_prob = 0.2
            if random.random() < congestion_prob:
                if interface == "FastEthernet":
                    bandwidth_reduction = random.uniform(0.3, 0.7)
                    latency_increase = random.uniform(20, 50)  # 延迟增加
                    packet_loss = random.uniform(5, 15)  # 丢包率
                elif interface == "GigabitEthernet":
                    bandwidth_reduction = random.uniform(0.2, 0.5)
                    latency_increase = random.uniform(10, 30)
                    packet_loss = random.uniform(2, 10)
                else:  # TenGigabitEthernet
                    bandwidth_reduction = random.uniform(0.1, 0.3)
                    latency_increase = random.uniform(5, 15)
                    packet_loss = random.uniform(0, 5)
                bandwidth = bandwidth * (1 - bandwidth_reduction)
                latency += latency_increase
                packet_loss = packet_loss

            # 10%概率模拟链路故障
            if random.random() < 0.1:
                ospf_state = 0
                bandwidth = bandwidth * 0.1  # 带宽降至10%
                latency += 100  # 延迟显著增加
                packet_loss += 20  # 丢包率增加

            bandwidth_utilization = random.uniform(10, 90)  # 10%-90%
            if random.random() < congestion_prob:
                bandwidth_utilization = min(bandwidth_utilization + random.uniform(20, 40), 100)  # 拥塞时利用率增加
            training_data["bandwidth_utilization"].append(round(bandwidth_utilization, 2))

            # 动态阈值，基于样本分布
            # 调整is_best评分，考虑拥塞影响
            score = (
                    0.35 * (local_pref / 200) +
                    0.25 * (1 - as_path_length / 5) +
                    0.15 * ospf_state +
                    0.15 * (bandwidth / bandwidth_options["TenGigabitEthernet"]) -
                    0.1 * (latency / 100) -
                    0.1 * (packet_loss / 100)
            )
            is_best = 1 if score > 0.6 + random.uniform(-0.1, 0.1) else 0

            training_data["is_best"].append(is_best)
            training_data["local_pref"].append(local_pref)
            training_data["bandwidth"].append(round(bandwidth))
            training_data["latency"].append(round(latency, 2))
            training_data["packet_loss"].append(round(packet_loss, 2))

        # 转换为DataFrame
        df = pd.DataFrame(training_data)

        # 数据验证
        assert len(df) == num_samples, "Length mismatch"
        assert df["as_path_length"].min() >= 1, "Invalid as_path_length"
        assert df["local_pref"].min() >= 0, "Invalid local_pref"
        assert set(df["ospf_state"]).issubset({0, 1}), "Invalid ospf_state"
        assert df["bandwidth"].min() >= 1, "Invalid bandwidth"
        assert set(df["is_best"]).issubset({0, 1}), "Invalid is_best"
        assert df["latency"].min() >= 0, "Invalid latency"
        assert df["packet_loss"].min() >= 0, "Invalid packet_loss"

        # 检查数据分布
        logging.info(f"Generated data summary:\n{df.describe()}")
        logging.info(f"is_best distribution: {df['is_best'].value_counts(normalize=True)}")

        df["path_cost"] = (
                0.5 * df["as_path_length"] +
                np.log1p(1000 / df["bandwidth"]) +  # 对倒数部分取对数，平滑分布
                (1 - df["ospf_state"]) * 0.2 +
                0.1 * df["latency"] +
                0.1 * df["packet_loss"] +
                0.05 * df["bandwidth_utilization"]  # 高利用率增加代价
        )

        logging.info(f"path_cost summary: mean={df['path_cost'].mean():.4f}, std={df['path_cost'].std():.4f}")

        return df

    def real_collect_data(self, device=None):
        """单步采集过程"""

        if not device:
            device = self.device

        # 收集真实OSPF邻居信息
        connection = ConnectHandler(**device)
        ospf_output = connection.send_command("show ip ospf neighbor")
        bgp_output = connection.send_command("show ip bgp")
        connection.disconnect()
        # 数据转换
        ospf_neighbors = self.parse_ospf_output(ospf_output)  # 自定义解析函数
        bgp_paths = self.parse_bgp_output(bgp_output)  # 自定义解析函数

        # 转换为训练集和测试集
        # training_data = self.generate_training_data(ospf_neighbors, bgp_paths)

        return {"ospf": ospf_neighbors, "bgp": bgp_paths}

    def main(self):
        if self.mode == "virtual":
            with open("./data/ospf_data.txt", "r") as f:
                ospf_output = f.read()

            with open("./data/bgp_data.txt", "r") as f:
                bgp_output = f.read()

            # 解析数据（示例）
            ospf_neighbors = self.parse_ospf_output(ospf_output)  # 自定义解析函数
            bgp_paths = self.parse_bgp_output(bgp_output)  # 自定义解析函数

            # 模拟n条训练数据
            training_data = self.Simulation_training_data(
                ospf_neighbors, bgp_paths,
                self.mode_set["virtual"]["num_samples"] if self.mode else 50
            )

            return training_data

        else:
            # 多设备批量采集
            with Pool(processes=2) as pool:
                results = pool.map(self.real_collect_data, self.device)

            return results["ospf"], results["bgp"]


if __name__ == '__main__':
    demo = Collect_network_data(mode="virtual", mode_set={"virtual": {"num_samples": 100}})
    training_data = demo.main()
    print(training_data)