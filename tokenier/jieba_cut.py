import encodings
from multiprocessing import context

import jieba
from numpy.char import isdigit
from tqdm import tqdm
import json
import pandas as pd
from  tinydb import TinyDB ,Query
from core.config import VOCAB_PATH,CSV_DIR,STOP_LIST,CONTENT_INFO
import os
import re
import string
from collections import Counter


URL_PATTERN = re.compile(
    r"https?://[^\s]+|www\.[^\s]+",
    re.IGNORECASE
)


PUNCS = set(string.punctuation) | {

    '，','。','！','？','；','：','、','（','）','《','》','【','】',
    '“','”','‘','’','—','…','～','·'
}


def read_csv(csv_file):
    csv_path=os.path.join(CSV_DIR,csv_file)
    df=pd.read_csv(csv_path,encoding='utf-8')
    return df

def load_stopwords(stop_path:str=None):
    with open(STOP_LIST,'r',encoding='utf-8') as f:
        stop_words=f.read()
        temp=''
        stop_list=[]
        for i,letter in enumerate(stop_words):
            if letter !='\n':
                temp+=letter
            else: 
                stop_list.append(temp)
                temp=''
    return stop_list

def info_write(code,content_list,info:TinyDB,time) :
   
    ws= dict(Counter(content_list))
    info.insert({'id':code,'words':ws,"time": time})
    return
    

def cut_process(data_file, df, vocab: TinyDB, exec: Query, info: TinyDB):
    word_counter = Counter()   # 内存累积词频
    docs_info = []             # 批量保存文档信息

    for _, row in tqdm(df.iterrows(), desc=f"数据表：{data_file}"):
        content = row['微博正文']
        code = row['微博id']
        time = row["发布时间"]

        content = URL_PATTERN.sub("", content)
        content_list = word_cut(content)

        # 累积词频
        word_counter.update(content_list)

        # 保存文档信息
        docs_info.append({'id': code, 'words': dict(Counter(content_list)), 'time': time})

    # 批量写入文档信息
    info.insert_multiple(docs_info)

    # 批量更新词表
    for word, freq in word_counter.items():
        res = vocab.get(exec.word == word)
        if res:
            vocab.update({'freq': res['freq'] + freq}, exec.word == word)
        else:
            vocab.insert({'word': word, 'freq': freq, 'lang': 'ch'})

stops=set(load_stopwords())


def vocab_create(words_list,vocab:TinyDB,exec: Query):
    for w in words_list:
        res=vocab.get(w==exec.word)
        if res :
           vocab.update({"freq" :res['freq']+1},w==exec.word)
        else :
            vocab.insert({"word":w, "freq":1,"lang":"ch"})

def word_cut(context:str):
    token_list=jieba.lcut(context)
    words_list=[]
    for word in token_list:
        word=word.strip().lower()
        if word in stops:
            continue
        elif word in PUNCS:
            continue
        elif word.isdigit():
            continue
        elif not word:
            continue
        elif  len(word) <= 1:
            continue
        else:
            words_list.append(word)
    return  words_list


def main():
    vocab=TinyDB(VOCAB_PATH)
    exec=Query()
    content=TinyDB(CONTENT_INFO)
    
    if  not isinstance(vocab,TinyDB):
        print("数据格式出错")
        return 
    print(os.listdir(CSV_DIR))
    for data_file in os.listdir(CSV_DIR):
        print(data_file)
        df=read_csv(data_file)
        cut_process(data_file,df,vocab,exec,content)

if __name__ == "__main__":
    main()