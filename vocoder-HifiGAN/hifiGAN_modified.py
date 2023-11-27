from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from hifiGAN_model import Generator

h = None
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


################## Modified ##################
def create_model(config_file, ckpt_path):
    # Load config File
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    generator = Generator(h).to(device)

    # Load checkpoint File
    state_dict_g = load_checkpoint(ckpt_path, device)
    generator.load_state_dict(state_dict_g['generator'])
    return generator

def inference_modified(generator, c=None):
    MAX_WAV_VALUE = 32768.0
    x = c
    x = torch.FloatTensor(x).to(device)
    y_g_hat = generator(x)
    audio = y_g_hat.squeeze()
    audio = audio * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype('int16')
    return audio


