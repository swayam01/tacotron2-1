import numpy as np
import re,os

phonemes = np.loadtxt('text/phoneme.lst','str')
tones    = range(8) #八种音调
phoneme2id = {ph:i for i,ph in enumerate(phonemes)}
tone2id  = {'0'   : 0, # 轻声
            '1'   : 1, 
            '2'   : 2, 
            '3'   : 3, 
            '4'   : 4, 
            '6'   : 5, 
            '7'   : 5,
            '8'   : 5,
            '9'   : 5,
            'XX'  : 6,  # 短静音 sp
            'sil' : 7} # 首尾静音sil

extract_phoneme = re.compile(r'.*-(.*)\+.*')
def text_to_sequence(filename):
    # 将fullab转为音素+音调的形式
    fulllab = np.loadtxt(os.path.join('data/fulllab/', filename+'.lab'), dtype='str')
    lis = []
    for line in fulllab:
        p = re.sub(extract_phoneme,'\\1',line)
        t = line[line.find('@')+1:line.find('$')]
        if len(t) > 2:
            t = 'sil'
        lis.append([phoneme2id[p], tone2id[t]])
    return lis