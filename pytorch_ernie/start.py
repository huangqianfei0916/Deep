"""
Copyright (c) 2024 by huangqianfei@tju.edu.cn All Rights Reserved. 
Author: huangqianfei@tju.edu.cn
Date: 2024-08-31 22:28:59
Description: 
"""
from tqdm import tqdm
from utils import logger
from pathlib import Path
from datetime import datetime  

import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from transformers import BertTokenizer, BertConfig, WarmUp
from torch.utils.data import DataLoader


from ernie import ErnieEncode
from data_tools import Reader

gpu = torch.cuda.is_available()
print("GPU available: ", gpu)
print("CuDNN: ", torch.backends.cudnn.enabled)
print('GPUs：', torch.cuda.device_count())

class FXBTask:

    def __init__(self, model_path="", save_dir=""):
        """init"""
        self.save_dir = save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        config = BertConfig.from_pretrained(model_path)


        self.model = ErnieEncode.from_pretrained(
            model_path,
            config,
            num_classes1=len(Reader.label1),
            num_classes2=len(Reader.label2),
        )
        
        logger.info(f" tokenizer, model loaded from {model_path} ok")


    def train_batch(self, model, batch):
        (input_ids, labels1, labels2) = batch
        (logits1, logits2) = model(input_ids=input_ids)
        loss1 = torch.nn.functional.cross_entropy(logits1, labels1)
        loss2 = torch.nn.functional.cross_entropy(logits2, labels2)
        total_loss = loss1 + loss2
        return total_loss, loss1, loss2

    def train(self, train_path, n_epoch=1, max_length=64, batch_size=128, learning_rate=3e-5):        
        self.model.train()

        dataset = Reader(
            train_path,
            tokenizer=self.tokenizer,
            max_token=max_length,
            shuffle=False,
        )
        dataloader = DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            shuffle=True,
            batch_size=12,
        )

        training_steps = len(dataloader) * n_epoch
        lr_scheduler = WarmUp(initial_learning_rate=learning_rate, warmup_steps=training_steps)

        optimizer = torch.optim.AdamW(
            lr=lr_scheduler,
            params=self.model.parameters(),
            weight_decay=0.01,
        )

        step = 0
        self.save(self.model, self.tokenizer, name=f"/step-{step}")
        progress_bar = tqdm(range(training_steps))
        for epoch in range(n_epoch):
            dataset = Reader(
                train_path,
                tokenizer=self.tokenizer,
                max_token=max_length,
                shuffle=False,
            )
            dataloader = DataLoader(
                dataset,
                collate_fn=dataset.collate_fn,
                return_list=True,
                shuffle=False,
                batch_size=batch_size,
                drop_last=True,
                num_workers=1,
            )
            for i, batch in enumerate(dataloader):
                if step % 100 == 0:
                    self.valid(batch)

                losses = self.train_batch(self.model, batch)
                total_loss, loss1, loss2 = losses
                total_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()

                (total_loss, loss1, loss2) = (x.item() for x in losses)

                line = f" lr:{lr_scheduler.get_lr():.2e} total_loss:{total_loss:.3f} loss1:{loss1} loss2:{loss2}  "
                progress_bar.update(1)
                progress_bar.set_description(line)
                step += 1
        
                if step % 100 == 0:
                    line = f"epoch:{epoch} step:{step} " + line
                    logger.info(line)


            self.save(self.model, self.tokenizer, name=f"/epoch-{epoch}")
            line = f"epoch:{epoch} step:{step} " + line
            logger.info(line)


        self.save(self.model, self.tokenizer, name=f"/epoch-{epoch}")
        logger.info("trained")

    def save(self, model, tokenizer, name):
        try:
            saved_dir = self.save_dir + name
            Path(saved_dir).mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(saved_dir)
            model.save_model_config(saved_dir)
            model.save_pretrained(saved_dir)
            logger.info(f" name:{name} saved -> {saved_dir}")

        except Exception as e:
            logger.error((e))

    def valid(self, batch):
        """valid"""
        (input_ids, labels1, labels2) = batch
        model = self.model
        model.eval()
        out_labels1 = labels1.detach().numpy()
        out_labels2 = labels2.detach().numpy()
  
        (logits1, logits2) = model(input_ids=input_ids)

        preds1 = np.argmax(logits1.detach().numpy(), axis=1).reshape([len(logits1), 1])
        preds2 = np.argmax(logits2.detach().numpy(), axis=1).reshape([len(logits2), 1])
        p, r, f, _ = metrics.precision_recall_fscore_support(out_labels1, preds1)
        logger.info(f"valid p:{p} r:{r} f:{f}")
        p, r, f, _ = metrics.precision_recall_fscore_support(out_labels2, preds2)
        logger.info(f"valid p:{p} r:{r} f:{f}")
        model.train()


if __name__ == "__main__":

    model_path = "/Users/huangqianfei/.transformer/ernie-3.0-mini-zh"
    now = datetime.now()  

    task = FXBTask(
        model_path=model_path,
        save_dir="model_checkpoint",
    )
    task.train(
        train_path="/Users/huangqianfei/workspace/learn_nlp/ernie/data/train_data.txt",
        n_epoch=1,
        max_length=32,
        batch_size=64,
    )
