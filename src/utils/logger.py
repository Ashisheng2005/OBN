#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2025/4/17 下午8:15 
# @Author : Huzhaojun
# @Version：V 1.0
# @File : logger.py
# @desc : README.md

import logging
import os
from src.utils.config import Config


def setup_logger(config: Config):
    """集中日志设置，实现跨模块的一致日志记录。"""

    log_level = config.get_nested('logging', 'level', default='INFO')
    log_file = config.get_nested('logging', 'file', default='./logs/app.log')

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)
