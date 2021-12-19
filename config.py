# -*- coding: utf-8 -*-

""" 
@author: Jin.Fish
@file: config.py
@version: 1.0
@time: 2021/12/18 17:17:39
@contact: jinxy@pku.edu.cn

config
"""

from dataclasses import dataclass


@dataclass
class Config:
    # train
    epoch_num: int = 5
    learning_rate: float = 1e-6
    report_step: int = 500
    require_improvement: int = 10000
    batch_size: int = 8

    # model
    dropout: float = 0.5

    # I/O
    log_dir: str = "result/train/demo/log"
    model_path: str = "result/train/demo/best.bin"

    # evaluate
    labels = ["False", "True"]
