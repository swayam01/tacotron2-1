import numpy as np
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))

phone_set = np.loadtxt(os.path.join(cur_dir, 'phoneme.lst'),'str')  # 61种phoneme

phone2id = {ph:i for i,ph in enumerate(phone_set)}

tone2id  = {'0'   : 0, # 轻声
            '1'   : 1, 
            '2'   : 2, 
            '3'   : 3, 
            '4'   : 4, 
            '6'   : 5, 
            '7'   : 5,
            '8'   : 5,
            '9'   : 5,
            'XX'  : 6,  # 短静音   sp
            'sil' : 7}  # 首尾静音 sil
