import numpy as np
from scipy.io.wavfile import read
import torch,os


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    #CPU version
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len))
    #ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).byte()
    return mask


def load_wav_to_torch(full_path, sr):
    sampling_rate, data = read(full_path)
    assert sr == sampling_rate, "{} SR doesn't match {} on path {}".format(
        sr, sampling_rate, full_path)
    return torch.FloatTensor(data.astype(np.float32))


def load_fbs_and_fb_text_dict(filename, text_path_root):
    fbs = np.loadtxt(filename, 'str')
    fb_text_dict = {}   
    for fb in fbs:
        text_path = os.path.join(text_path_root, fb+'.lab')
        text = np.loadtxt(text_path, 'str')
        fb_text_dict[fb] = text
    return fbs, fb_text_dict


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
