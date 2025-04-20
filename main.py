#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2025/4/17 下午8:12 
# @Author : Huzhaojun
# @Version：V 1.0
# @File : main.py
# @desc : README.md

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.data_collection.collector import NetworkDataCollector
from src.preprocessing.data_processor import DataProcessor
from src.training.trainer import Trainer


def main():
    """协调整个管道"""

    config = Config('config/config.yaml')
    logger = setup_logger(config)

    try:
        logger.info("Starting network path optimization pipeline")

        # Collect data
        collector = NetworkDataCollector(config)
        raw_data = collector.collect()

        # Process data
        processor = DataProcessor(config)
        training_data = processor.simulate_training_data(raw_data['ospf'], raw_data['bgp'])

        # Train models
        trainer = Trainer(config, training_data)
        trainer.train()

        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == '__main__':
    main()
