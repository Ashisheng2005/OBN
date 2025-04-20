#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2025/4/17 下午8:15 
# @Author : Huzhaojun
# @Version：V 1.0
# @File : test_parser.py
# @desc : README.md

import unittest
from src.data_collection.parser import OSPFParser
from src.utils.config import Config


class TestOSPFParser(unittest.TestCase):
    """验证解析逻辑的样例单元测试"""

    def setUp(self):
        self.config = Config('config/config.yaml')
        self.parser = OSPFParser(self.config)

    def test_parse_valid_output(self):
        sample_output = """
        *  192.168.1.0/24  10.0.0.1  100  100  0  65001  i
        """
        routes = self.parser.parse(sample_output)
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0]['network'], '192.168.1.0/24')
        self.assertEqual(routes[0]['next_hop'], '10.0.0.1')


if __name__ == '__main__':
    unittest.main()
