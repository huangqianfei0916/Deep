"""
Copyright (c) 2024 by huangqianfei@tju.edu.cn All Rights Reserved. 
Author: huangqianfei@tju.edu.cn
Date: 2024-08-31 22:28:59
Description: 
"""
import os
from pathlib import Path
from datetime import datetime  
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from transformers import BertTokenizer, ErnieModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  

from ernie import ErnieEncode
from data_tools import Reader
from utils import logger
from utils import LinearWarmupCosineAnnealingLR

gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU available: ", gpu)
print("CuDNN: ", torch.backends.cudnn.enabled)
print("GPUs: ", torch.cuda.device_count())
print("device: ", device)

class FXBTask:

    def __init__(self, model_path="", save_dir=""):
        """init"""
        self.save_dir = save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = ErnieModel.from_pretrained(
            model_path
        )
        self.multi_task_model = ErnieEncode(
            self.model,
            num_classes1=len(Reader.label1),
            num_classes2=len(Reader.label2),
        )
        
        logger.info(f" tokenizer, model loaded from {model_path} ok")


    def train_batch(self, model, batch, criterion1, criterion2):
        input_ids, labels1, labels2, attention_mask = batch
        input_ids, labels1, labels2 = input_ids.to(device), labels1.to(device), labels2.to(device)
        attention_mask = attention_mask.to(device)
        
        logits1, logits2 = model(input_ids=input_ids, attention_mask=attention_mask)

        loss1 = criterion1(logits1, labels1)
        loss2 = criterion2(logits2, labels2)

        total_loss = loss1 + loss2
        return total_loss, loss1, loss2

    def train(self, train_path, n_epoch=5, max_length=32, batch_size=128, learning_rate=3e-5):        
        self.multi_task_model.train()

        data_set = Reader(
            train_path,
            tokenizer=self.tokenizer,
            max_token=50,
            shuffle=True,
        )
        dataloader = DataLoader(
            data_set,
            collate_fn=data_set.collate_fn,
            shuffle=True,
            batch_size=64,
        )
        swriter = SummaryWriter(os.path.join(self.save_dir, 'log'))
        

        training_steps = len(dataloader) * n_epoch
        print("training_steps:", training_steps)

        optimizer = torch.optim.AdamW(
            lr=learning_rate,
            params=self.multi_task_model.parameters(),
            weight_decay=0.01,
        )

        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=1, total_epochs=n_epoch, cycles=0.5)  
        criterion1 = nn.CrossEntropyLoss()  
        criterion2 = nn.BCELoss()

        step = 0
        progress_bar = tqdm(range(training_steps))
        self.multi_task_model.to(device)
        for epoch in range(n_epoch):

            data_set = Reader(
                train_path,
                tokenizer=self.tokenizer,
                max_token=50,
                shuffle=True,
            )
            dataloader = DataLoader(
                data_set,
                collate_fn=data_set.collate_fn,
                shuffle=True,
                batch_size=64,
            )
            for i, batch in enumerate(dataloader):
                if step % 50 == 0:
                    self.valid(batch)

                losses = self.train_batch(self.multi_task_model, batch, criterion1, criterion2)
                total_loss, loss1, loss2 = losses
                swriter.add_scalar('loss', total_loss.cpu().detach().numpy(), step)
                swriter.add_scalar('loss1', loss1.cpu().detach().numpy(), step)
                swriter.add_scalar('loss2', loss2.cpu().detach().numpy(), step)
                swriter.add_scalar('lr', np.array(lr_scheduler.get_lr()), step)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                lr_scheduler.step()

                (total_loss, loss1, loss2) = (x.item() for x in losses)

                line = f" lr:{lr_scheduler.get_lr()} total_loss:{total_loss:.3f} loss1:{loss1} loss2:{loss2}  "
                progress_bar.update(1)
                progress_bar.set_description(line)
                step += 1
        
                if step % 100 == 0:
                    line = f"epoch:{epoch} step:{step} " + line
                    logger.info(line)


            self.save(self.multi_task_model, self.tokenizer, name=f"/epoch-{epoch}")
            line = f"epoch:{epoch} step:{step} " + line
            logger.info(line)
            swriter.close()


        self.save(self.multi_task_model, self.tokenizer, name=f"/epoch-{epoch}")
        logger.info("trained")

    def save(self, model, tokenizer, name):
        try:
            saved_dir = self.save_dir + name
            
            state_path = saved_dir + "/state"
            Path(state_path).mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(state_path)
            torch.save(model.state_dict(), state_path + "/model_state.pt")
            torch.save(model, state_path + "/model_all.pt")
            logger.info(f" name:{name} saved -> {state_path}")

            
            # jit_path = saved_dir + "/jit"
            # Path(jit_path).mkdir(parents=True, exist_ok=True)
            # scripted_module = torch.jit.script(model)  
            # torch.jit.save(scripted_module, jit_path + "/scripted_model.pt")

        except Exception as e:
            logger.error((e))

    def valid(self, batch):
        """valid"""
        input_ids, labels1, labels2, attention_mask = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        model = self.multi_task_model
        model.eval()
        logits1, logits2 = model(input_ids=input_ids, attention_mask=attention_mask)
        
        preds1 = np.argmax(logits1.cpu().detach().numpy(), axis=1).reshape([len(logits1), 1])
        acc = metrics.accuracy_score(labels1, preds1)   
        report = metrics.classification_report(labels1, preds1)
        print(report)
        logger.info(f"task1 valid acc:{acc}")

        logits2 = logits2.cpu().detach().numpy()
        logits2 = (logits2 > 0.5).astype(int)  

        acc = metrics.accuracy_score(labels2, logits2)   
        report = metrics.classification_report(labels2, logits2)
        print(report)
        logger.info(f"task2 valid acc:{acc}")
        model.train()

if __name__ == "__main__":

    model_path = "/Users/huangqianfei/.transformer/ernie-3.0-mini-zh"
    now = datetime.now()  

    task = FXBTask(
        model_path=model_path,
        save_dir="model_checkpoint",
    )
    task.train(
        train_path="/Users/huangqianfei/workspace/learn_nlp/pytorch_ernie/data/train_data.txt",
        n_epoch=5,
        max_length=32,
        batch_size=64,
    )

