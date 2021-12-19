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
    learning_rate: float = 2e-5
    report_step: int = 200
    require_improvement: int = 10000
    batch_size: int = 4

    # task
    choice_num: int = 5

    # model
    dropout: float = 0.5
    riddle_max_len: int = 32
    choice_max_len: int = 128
    hidden_size: int = 256

    # I/O
    log_dir: str = "result/train/demo/log"
    model_path: str = "result/train/demo/best.bin"

    # evaluate
    labels = ["False", "True"]
