import os
import logging
import glob
import torch  
import math
from torch.optim.lr_scheduler import _LRScheduler  
from pathlib import Path


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

def save_model(model, tokenizer, optimizer,step, model_path):
    try:

        Path(model_path).mkdir(parents=True, exist_ok=True)
        
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
        }
        torch.save(state, model_path + "/model_state.pt")
        tokenizer.save_pretrained(model_path)
        logger.info(f" model saved -> {model_path}")

    except Exception as e:
        logger.error((e))

        
def load_model(model, optimizer, model_path):
    filename = model_path + "/model_state.pt"
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']    

    return model, optimizer, step

    
def update_version(model_path):
    def version(x):
        """version"""
        x = int(x.split("-")[-1])
        return x

    ckpt_paths = glob.glob(os.path.join(model_path, "model-*"))

    output_dir = ""
    if len(ckpt_paths) > 0:
        output_dir = sorted(ckpt_paths, key=version, reverse=True)[0]
    
    return output_dir


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """warm up"""
    def __init__(self, optimizer, warmup_steps, total_steps, initial_lr=5e-5, min_lr=0.0, cycles=1, last_epoch=-1, verbose=False):
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

            
if __name__ == "__main__":
    model_path = "/Users/huangqianfei/workspace/learn_nlp/pytorch_ernie/model_checkpoint/"
    output_dir = update_version(model_path)
    print(output_dir)