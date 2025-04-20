#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2025/4/17 下午8:15 
# @Author : Huzhaojun
# @Version：V 1.0
# @File : config.py
# @desc : README.md

import yaml
import os


class Config:
    """加载YAML配置文件并提供对设置的访问。"""

    def __init__(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} not found.")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def get_nested(self, *keys, default=None):
        value = self.config
        for key in keys:
            value = value.get(key, default)
            if value is default:
                return default
        return value

