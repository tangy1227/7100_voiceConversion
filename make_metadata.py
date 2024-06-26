"""
Generate speaker embeddings and metadata for training
speaker/style encoder part
"""
import os
import pickle
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch

import torchaudio
from speechbrain.pretrained import EncoderClassifier

from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path

#############################################
#                 D-vector                  #
#############################################
speaker_encoder_model = '/home/ytang363/7100_voiceConversion/pretrain/3000000-BL.ckpt'
C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
c_checkpoint = torch.load(speaker_encoder_model) # model_b, model_l, optimizer
# print(c_checkpoint['model_b'])
new_state_dict = OrderedDict()
for key, val in c_checkpoint['model_b'].items():
    new_key = key[7:]
    new_state_dict[new_key] = val
C.load_state_dict(new_state_dict)

len_crop = 128 
tmp = np.load('/home/ytang363/7100_voiceConversion/VCTK-Corpus-0.92/spmel-16k/p225/p225_002_mic1.npy') # (129, 80)
left = np.random.randint(0, tmp.shape[0]-len_crop)
melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda() # [1, len, 80]
emb = C(melsp) # [1, 256]

#############################################
#                speechbrain                #
#############################################
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
signal, fs = torchaudio.load('/home/ytang363/7100_voiceConversion/VCTK-Corpus-0.92/wav-16k/p226/p226_003_mic1.wav')
embeddings = classifier.encode_batch(signal)
embeddings = torch.squeeze(embeddings, dim=1)
print(embeddings.shape)

#############################################
#                resemblyzer                #
#############################################
fpath = Path("/home/ytang363/7100_voiceConversion/VCTK-Corpus-0.92/wav-16k/p226/p226_003_mic1.wav")
wav = preprocess_wav(fpath)

encoder = VoiceEncoder()
embed = encoder.embed_utterance(wav)
np.set_printoptions(precision=3, suppress=True)
print(embed.shape)


##########################################################################################
# Directory containing mel-spectrograms
# rootDir = '/home/ytang363/7100_voiceConversion/VCTK-Corpus-0.92/spmel-16k-split/test'    # melspec
rootDir = '/home/ytang363/7100_voiceConversion/VCTK-Corpus-0.92/wav-16k-split/validation'        # wavFile
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

num_uttrs = 10
len_crop = 128
speakers = []
count = 0
spk_emb_name = 'resemblyzer' # dvec, xvec, resemblyzer

for speaker in sorted(subdirList):

    print('Processing speaker: %s' % speaker)
    utterances = []
    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
    
    # make speaker embedding
    assert len(fileList) >= num_uttrs
    idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
    embs = []
    for i in range(num_uttrs):
        filePath = os.path.join(dirName, speaker, fileList[idx_uttrs[i]])
        
        ###### D-vector Model ######
        if spk_emb_name == 'dvec':
            tmp = np.load(filePath)
            candidates = np.delete(np.arange(len(fileList)), idx_uttrs)
            
            # choose another utterance if the current one is too short
            while tmp.shape[0] < len_crop:
                idx_alt = np.random.choice(candidates)
                tmp = np.load(os.path.join(dirName, speaker, fileList[idx_alt]))
                candidates = np.delete(candidates, np.argwhere(candidates==idx_alt))

            if tmp.shape[0]-len_crop == 0:
                left = 0
            else:
                left = np.random.randint(0, tmp.shape[0]-len_crop)
            melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda()

            emb = C(melsp)
            embs.append(emb.detach().squeeze().cpu().numpy())

        ###### Speechbrain ######
        if spk_emb_name == 'xvec':
            candidates = np.delete(np.arange(len(fileList)), idx_uttrs)

            signal, fs = torchaudio.load(filePath)
            ## Protential bug fix here
            while len(signal[0]) < (fs): # less than a second 
                idx_alt = np.random.choice(candidates)
                signal, fs = torchaudio.load(os.path.join(dirName, speaker, fileList[idx_alt]))
                candidates = np.delete(candidates, np.argwhere(candidates==idx_alt))

            emb = classifier.encode_batch(signal)
            emb = torch.squeeze(emb, dim=1) 
            embs.append(emb.detach().squeeze().cpu().numpy())    

        ###### resemblyzer ######
        if spk_emb_name == 'resemblyzer':
            encoder = VoiceEncoder()
            candidates = np.delete(np.arange(len(fileList)), idx_uttrs)

            wav = preprocess_wav(filePath)
            # signal, fs = torchaudio.load(filePath)
            while len(signal[0]) < (fs): # less than a second
                idx_alt = np.random.choice(candidates)
                wav = preprocess_wav(os.path.join(dirName, speaker, fileList[idx_alt]))
                candidates = np.delete(candidates, np.argwhere(candidates==idx_alt))

            emb = encoder.embed_utterance(wav)
            embs.append(emb)                           
    
    utterances.append(np.mean(embs, axis=0))
    
    # create file list
    for fileName in sorted(fileList):
        if fileName.split('.')[-1] != 'npy':
            fileName = fileName.split('.')[0] + '.npy'
        utterances.append(os.path.join(speaker,fileName))
    speakers.append(utterances)

# spmel/train.pkl
print(os.path.join(rootDir, 'train_res.pkl'))
with open(os.path.join(rootDir, 'train_res.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)