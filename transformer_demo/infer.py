
import torch
from tqdm import tqdm

import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Translator import Translator

from transformers import BertTokenizer
from utils import load_model

class InferData(object):
    def __init__(self, model_path):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Transformer(
                n_src_vocab=39980, 
                n_trg_vocab=39980,
                src_pad_idx=0, 
                trg_pad_idx=0)

        checkpoint = torch.load(model_path + "/model_state.pt")
        self.model.load_state_dict(checkpoint['model_state_dict']) 

        self.model.eval()
        self.model.to(self.device)

    def infer(self, line):
        '''Main Function'''
        src_pad_idx = 0
        trg_pad_idx = 0
        trg_bos_idx = 1
        trg_eos_idx = 2
        unk_idx = 39978
        
        translator = Translator(
            self.model,
            beam_size=2,
            max_seq_len=60,
            src_pad_idx=src_pad_idx,
            trg_pad_idx=trg_pad_idx,
            trg_bos_idx=trg_bos_idx,
            trg_eos_idx=trg_eos_idx).to(self.device)
        
        line = self.tokenizer.tokenize(line)
        src_seq = self.tokenizer.encode(line)
        src_seq = src_seq[1:]
        pred_seq = translator.translate_sentence(torch.LongTensor([src_seq]).to(self.device))

        pred_line = "".join(self.tokenizer.convert_ids_to_tokens(pred_seq))
        print(pred_line)

        print('[Info] Finished.')

if __name__ == "__main__":
    '''
    Usage: python translate.py -model trained.chkpt -data multi30k.pt -no_cuda
    '''
    model_path = "/Users/huangqianfei01/Desktop/learn/learn_nlp/transformer_demo/model_checkpoint/model-240/"
    infer_data = InferData(model_path)
    infer_data.infer("项目初审的申报条件是什么？")
