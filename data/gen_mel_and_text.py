import sys,os,re
sys.path.append('..')

import layers
from hparams import create_hparams
from glob import glob
from utils import load_wav_to_torch
import torch
import numpy as np
from tqdm import tqdm

def FileBaseName(filename):
    return os.path.basename(filename).split('.')[0]

def SaveMkdir(dir):
    try:
        if not os.path.exists(dir):
            os.mkdir(dir)
    except:
        os.makedirs(dir)

# 针对目录计算均值数据
def cal_MeanStd(datadir, dim, ref_file=None):
    # This method is efficient for large datadir
    # First row is mean vector
    # Second row is std vector
    tqdm.write('Calculate MeanStd Mean File...')
    files = os.listdir(datadir) 
    if ref_file!=None: 
        ref_list = np.loadtxt(ref_file,'str')
        files = [file for file in files if file.split('.')[0] in ref_list]
    filenum = len(files)
    mean_std = np.zeros([2,dim],dtype=np.float64)
    file_mean = np.zeros([filenum,dim+1],dtype=np.float64)
    file_std = np.zeros([filenum,dim+1],dtype=np.float64)
    for i in tqdm(range(len(files))):     
        file = datadir+os.sep+files[i]
        data = np.load(file)
        file_mean[i][0] = data.shape[0]
        file_std[i][0] = data.shape[0]      
        file_mean[i][1:] = np.mean(data,0)
        file_std[i][1:] = np.mean(data**2,0)
    
    file_sum = (file_mean[:,0]*file_mean[:,1:].T).T 
    file_ssum = (file_std[:,0]*file_std[:,1:].T).T 
    
    mean_std[0] = np.sum(file_sum,0) / np.sum(file_mean[:,0])   
    mean_std[1] = np.sqrt(np.sum(file_ssum,0)/ np.sum(file_mean[:,0]) - mean_std[0]**2)
    return mean_std

if __name__ == "__main__":
    gen_mel = 1
    gen_text = 1

    hparams = create_hparams()
    stft = layers.TacotronSTFT(
                hparams.filter_length, hparams.hop_length, hparams.win_length,
                hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                hparams.mel_fmax)

    if gen_mel:
        audio_files = sorted(glob('audio/*.wav'))
        out_dir = 'mel'
        SaveMkdir(out_dir)
        for file in tqdm(audio_files):
            tqdm.write(file)
            file_basename = os.path.basename(file).split('.')[0]
            audio_path = os.path.join(hparams.audio_path, file_basename+'.wav')
            audio, sampling_rate = load_wav_to_torch(audio_path, hparams.sampling_rate)
            audio_norm = audio / hparams.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = stft.mel_spectrogram(audio_norm)
            #转置存错 即数据行代表帧 列代表特征
            melspec = torch.squeeze(melspec, 0).numpy().transpose()
            
            out_file = os.path.join(out_dir, file_basename+'.npy')
            np.save(out_file, melspec)

        mean_std = cal_MeanStd(out_dir,hparams.n_mel_channels, ref_file=None)
        np.save(os.path.join(out_dir, os.pardir, 'MeanStd_Tacotron_mel.npy'),mean_std)

    if gen_text:
        lab_files = sorted(glob('fulllab/*.lab'))
        out_dir = 'text'
        SaveMkdir(out_dir)
        extract_phoneme = re.compile(r'.*-(.*)\+.*')
        def _text_to_sequence(absolute_path):
            # 将fullab转为音素+音调的形式
            fulllab = np.loadtxt(absolute_path, dtype='str')
            lis = []
            for line in fulllab:
                p = re.sub(extract_phoneme,'\\1',line)
                t = line[line.find('@')+1:line.find('$')]
                if len(t) > 2:
                    t = 'sil'
                lis.append(p+'\t'+t)
            return lis

        for file in tqdm(lab_files):
            tqdm.write(file)
            file_basename = os.path.basename(file).split('.')[0]
            lab_path = os.path.join('fulllab', file_basename+'.lab')
            text_path = os.path.join('text', file_basename+'.lab')
            text = _text_to_sequence(lab_path)
            np.savetxt(text_path, text, fmt="%s")

