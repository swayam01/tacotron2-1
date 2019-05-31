# -*- coding: utf-8 -*-
import numpy as np
from scipy.io.wavfile import read
import torch,os


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    #源代码使用CUDA
    #ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).byte()
    return mask


def load_wav_to_torch(absolute_path):
    sampling_rate, data = read(absolute_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths(filename):
    return np.loadtxt(filename, 'str')


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
