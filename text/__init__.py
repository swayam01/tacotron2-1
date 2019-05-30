import numpy as np
import re,os

thisfile_pwd = os.path.dirname(os.path.abspath(__file__))
phonemes = np.loadtxt(os.path.join(thisfile_pwd, 'phoneme.lst'),'str')
tones    = ['3', '1', '0', 'XX', '2', '4', '7', '6', '9', '8']
phone2id = {ph:i for i,ph in enumerate(phonemes)}
tone2id  = {'XX':0, 
            '0' :6,
            '1' :1, 
            '2' :2, 
            '3' :3, 
            '4' :4, 
            '6' :5, 
            '7' :5,
            '8' :5,
            '9' :5}

def text_to_sequence(text):
  # 将fullab转为音素+音调的形式
  return [[phone2id[p], tone2id[tn]] for p,t in text]