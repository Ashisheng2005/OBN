#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2025/4/17 下午8:12 
# @Author : Huzhaojun
# @Version：V 1.0
# @File : parser.py
# @desc : README.md

import re
from src.utils.logger import setup_logger
from src.utils.config import Config


class RoutingParser:
    def __init__(self, config: Config):
        self.logger = setup_logger(config)

    def parse(self, output):
        raise NotImplementedError("Subclasses must implement parse method")


class OSPFParser(RoutingParser):
    """整合OSPF输出的解析逻辑"""

    def parse(self, output):
        routes = []
        lines = output.split("\n")
        data_start = False
        for line in lines:
            if not line.strip() or line.startswith("Status codes") or line.startswith("Origin codes"):
                continue
            if line.strip().startswith("*") or line.strip().startswith(">"):
                data_start = True
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
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Skipping unparsable OSPF line: {line} ({e})")
        return routes


class BGPParser(RoutingParser):
    """整合BGP输出的解析逻辑"""

    def parse(self, output):
        routes = []
        lines = output.split("\n")
        for line in lines:
            if not line.strip() or line.startswith("Status codes") or line.startswith("Origin codes"):
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
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Skipping unparsable BGP line: {line} ({e})")
        return routes