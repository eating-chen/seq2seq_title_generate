from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
from config import *
from preprocess import *
from seq2seq_model import *
from train import *
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import jieba.posseg as pos
import jieba
import csv


if __name__ == '__main__':
    jieba.set_dictionary(tradition_dict_path)
    msg_list=[]
    with open(csv_file_path, newline='') as csvfile:
        # 讀取 CSV 檔案內容
        rows = list(csv.reader(csvfile))
        print(len(rows))
        for row in rows:
            msg_list.append([row[0], row[1]])

    content_class, title_class, pairs = prepareData(msg_list)
    print(random.choice(pairs))
    encoder1 = EncoderRNN(content_class.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, title_class.n_words, dropout_p=0.1).to(device)
    trainIters(encoder1, attn_decoder1, 150000, pairs, content_class, title_class, print_every=3000)
    evaluateRandomly(content_class, title_class, pairs, encoder1, attn_decoder1)
