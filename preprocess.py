#-------------------------------
#將轉為符合seq2seq的格式
#-------------------------------

import jieba.posseg as pos
import jieba
import torch
from config import *
from et_class import *

def read_txt_list(filepath):  
    words = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return words

def filterPair(p, max_len):
    return len(p[0].split(' ')) < max_len and \
        len(p[1].split(' ')) < max_len

def filterPairs(pairs, max_len):
    return [pair for pair in pairs if filterPair(pair, max_len)]

def read_data(data, stop_word_list):
    print("Reading csv data...")
    pairs = []
    count = 0
    for d in data:
        count += 1
        content = d[0].lower()
        title = d[1].lower()
        content_list = []
        content_line = ''
        title_list = []
        title_line = ''
        content = content.replace('\n', '')
        content = content.replace(' ', '')
        seg_content = pos.cut(content)
        for seg in seg_content:
            if seg.flag != 'x' and seg.word not in stop_word_list:
                content_list.append(seg.word)
        content_line = ' '.join(content_list)
        seg_title = pos.cut(title)
        for seg in seg_title:
            if seg.flag != 'x' and seg.word not in stop_word_list:
                title_list.append(seg.word)
        title_line = ' '.join(title_list)
        pairs.append([content_line, title_line])
        if count%100 == 0:
            print('已完成', count)
    content_class = et_nlp_class('content')
    title_class = et_nlp_class('title')

    return content_class, title_class, pairs

def prepareData(data):
    stop_word_list = read_txt_list(stop_list_path)
    content_class, title_class, pairs = read_data(data, stop_word_list)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs, MAX_LENGTH)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        content_class.addSentence(pair[0])
        title_class.addSentence(pair[1])
    print("Counted words:")
    print(content_class.name, content_class.n_words)
    print(title_class.name, title_class.n_words)
    return content_class, title_class, pairs