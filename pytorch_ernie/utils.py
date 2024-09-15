"""
Copyright (c) 2024 by huangqianfei@tju.edu.cn All Rights Reserved. 
Author: huangqianfei@tju.edu.cn
Date: 2024-09-01 11:56:16
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
            
    def __init__(self, optimizer, warmup_steps, total_steps, initial_lr=3e-5, min_lr=0.0, cycles=1, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.cycles = cycles
        self.verbose = verbose
        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch, verbose)
        

    def get_lr(self):
        current_step = self.last_epoch + 1
        cycle_length = self.total_steps // self.cycles
        current_cycle = current_step // cycle_length
        current_cycle_step = current_step % cycle_length

        if current_cycle_step < self.warmup_steps:
            lr = self.initial_lr * (current_cycle_step / self.warmup_steps)
        else:
            progress = (current_cycle_step - self.warmup_steps) / (cycle_length - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        if self.verbose:
            print(f"Cycle {current_cycle}, Step {current_cycle_step}: Learning Rate = {lr}")
        
        return [lr for _ in self.base_lrs]