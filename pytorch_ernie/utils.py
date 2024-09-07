"""
Copyright (c) 2024 by huangqianfei@tju.edu.cn All Rights Reserved. 
Author: huangqianfei@tju.edu.cn
Date: 2024-09-01 09:02:35
Description: 
"""
import logging
import torch  
import math
from torch.optim.lr_scheduler import _LRScheduler  


def spider_log():
    logger = logging.getLogger('ernie_task')
    logger.setLevel(logging.DEBUG)

    # 创建一个handler，用于写log日志
    file_handler = logging.FileHandler('log/ernie_task.log')
    file_handler.setLevel(logging.INFO)

    # 创建一个handler，用于输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = spider_log()


class LinearWarmupCosineAnnealingLR(_LRScheduler):  
    def __init__(self, optimizer, warmup_epochs, total_epochs, cycles, last_epoch=-1):  
        self.warmup_epochs = warmup_epochs  
        self.total_epochs = total_epochs  
        self.cycles = cycles  
        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)  
  
    def get_lr(self):  
        if self.last_epoch < self.warmup_epochs:  
            # 线性warmup  
            return [base_lr * (float(self.last_epoch + 1) / self.warmup_epochs) for base_lr in self.base_lrs]  
        else:  
            # 余弦退火  
            completed_cycles = self.last_epoch // (self.total_epochs - self.warmup_epochs)  
            progress_within_cycle = (self.last_epoch - self.warmup_epochs) % (self.total_epochs - self.warmup_epochs)  
            progress_within_cycle = progress_within_cycle / (self.total_epochs - self.warmup_epochs)  
            return [base_lr * (0.5 * (1. + math.cos(math.pi * (completed_cycles + progress_within_cycle) / self.cycles)))  
                    for base_lr in self.base_lrs]  