#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2025/4/17 下午8:12 
# @Author : Huzhaojun
# @Version：V 1.0
# @File : collector.py
# @desc : README.md

from netmiko import ConnectHandler
from multiprocessing import Pool
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.data_collection.parser import OSPFParser, BGPParser


class NetworkDataCollector:
    """处理OSPF/BGP数据采集（实模式或虚模式）"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger(config)
        self.mode = config.get_nested('data_collection', 'mode', default='virtual')
        self.ospf_parser = OSPFParser(self.config)
        self.bgp_parser = BGPParser(self.config)

    def collect_virtual_data(self):
        try:
            ospf_path = self.config.get_nested('data_collection', 'virtual', 'ospf_data_path')
            bgp_path = self.config.get_nested('data_collection', 'virtual', 'bgp_data_path')
            with open(ospf_path, 'r') as f:
                ospf_output = f.read()
            with open(bgp_path, 'r') as f:
                bgp_output = f.read()
            ospf_data = self.ospf_parser.parse(ospf_output)
            bgp_data = self.bgp_parser.parse(bgp_output)
            return {'ospf': ospf_data, 'bgp': bgp_data}

        except Exception as e:
            self.logger.error(f"Failed to collect virtual data: {e}")
            raise

    def collect_real_data(self, device):
        try:
            connection = ConnectHandler(**device)
            ospf_output = connection.send_command("show ip ospf neighbor")
            bgp_output = connection.send_command("show ip bgp")
            connection.disconnect()
            ospf_data = self.ospf_parser.parse(ospf_output)
            bgp_data = self.bgp_parser.parse(bgp_output)
            return {'ospf': ospf_data, 'bgp': bgp_data}
        except Exception as e:
            self.logger.error(f"Failed to collect real data from {device['host']}: {e}")
            raise

    def collect(self):
        if self.mode == 'virtual':
            return self.collect_virtual_data()
        else:
            devices = self.config.get_nested('data_collection', 'real', 'devices', default=[])
            with Pool(processes=len(devices)) as pool:
                results = pool.map(self.collect_real_data, devices)
            return results

