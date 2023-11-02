"""
Generate speaker embeddings and metadata for testing,
which contains [filename, speaker embedding, spectrogram]
speaker embedding shape: (256,)
spectrogram shape: (frames, 80)
"""
import os
import pickle
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch

speaker_encoder_model = '/home/ytang363/7100_voiceConversion/pretrain/3000000-BL.ckpt'
C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
c_checkpoint = torch.load(speaker_encoder_model)
new_state_dict = OrderedDict()
for key, val in c_checkpoint['model_b'].items():
    new_key = key[7:]
    new_state_dict[new_key] = val
C.load_state_dict(new_state_dict)

num_uttrs = 10
len_crop = 128
rootDir = '/home/ytang363/7100_voiceConversion/VCTK-Corpus-0.92/spmel-16k'
dirName, subdirList, _ = next(os.walk(rootDir))
process_speakers = ['p225', 'p226', 'p227', 'p228', 'p229', 'p230', 'p231', 'p232', 'p233', 'p234', 'p237']
process_uttr = '002'
speakers = []

for speaker in sorted(subdirList):
    if speaker not in process_speakers:
        continue

    print('Processing speaker: %s' % speaker)
    utterances = []

    #### Add first element to utterances ####
    utterances.append(speaker) 
    _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
    
    #### speaker embedding ####
    assert len(fileList) >= num_uttrs # test to see if data set utterance is greater than the minimum
    idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False) # randomly pick number of utterance for each speaker in VCTK
    embs = []
    for i in range(num_uttrs):
        tmp = np.load(os.path.join(dirName, speaker, fileList[idx_uttrs[i]]))
        candidates = np.delete(np.arange(len(fileList)), idx_uttrs) # remove the idx_uttrs and create a new array with size of [speaker uttr - len(idx_uttrs)]

        # choose another utterance if the current one is too short
        while tmp.shape[0] <= len_crop:
            idx_alt = np.random.choice(candidates)
            tmp = np.load(os.path.join(dirName, speaker, fileList[idx_alt]))
            candidates = np.delete(candidates, np.argwhere(candidates==idx_alt))

        left = np.random.randint(0, tmp.shape[0]-len_crop)
        melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda() # shape: [1, 128, 80]

        # D_VECTOR Model
        emb = C(melsp) # shape: (1, 256)
        embs.append(emb.detach().squeeze().cpu().numpy()) # shape: (10, 256), and after taking the mean, it becomes (256, )

    #### Add Second element to utterances ####
    utterances.append(np.mean(embs, axis=0)) # utterances = ['speaker', embs: (256, )]

    #### Add Third element to utterances ####
    idx_uttrs_spec = np.random.choice(len(fileList), size=1, replace=False)[0]
    indices = [i for i, element in enumerate(fileList) if process_uttr in element]

    # file = fileList[idx_uttrs_spec]
    file = fileList[indices[0]]
    spec = np.load(os.path.join(dirName, speaker, file))
    print(f'file name: {file}, shape: {spec.shape}')
    utterances.append(spec)

    speakers.append(utterances)

path_dir = '/home/ytang363/7100_voiceConversion'
print(os.path.join(path_dir, 'metadata_002.pkl'))
with open(os.path.join(path_dir, 'metadata_002.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)
              

