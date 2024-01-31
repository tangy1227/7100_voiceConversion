import os
import argparse
import torch
from solver_encoder import Solver
from dataLoader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Data loader.
    # Get Data from spmel/ (converted melspec from wavs/)
    # orginal size torch.Size([2, 128, 80]), torch.Size([2, 256])
    # xvector size torch.Size([2, 128, 80]), torch.Size([2, 512])
    vcc_loader = get_loader(config.data_dir, config.batch_size, config.len_crop)
    
    solver = Solver(vcc_loader, config)

    solver.train()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--lambda_cd', type=float, default=1, help='weight for hidden code loss')
    parser.add_argument('--dim_neck', type=int, default=32)
    parser.add_argument('--dim_emb', type=int, default=512) # 256
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=32)
    
    # Training configuration.
    parser.add_argument('--data_dir', type=str, default='/home/ytang363/7100_voiceConversion/VCTK-Corpus-0.92/spmel-16k') # ./spmel
    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=1500000, help='number of total iterations')
    parser.add_argument('--len_crop', type=int, default=128, help='dataloader output sequence length')
    
    # Miscellaneous.
    parser.add_argument('--log_step', type=int, default=100)

    # Tensorboard.
    parser.add_argument('--log_dir', type=str, default='/home/ytang363/7100_voiceConversion/logs')
    parser.add_argument('--num_ckpt', type=int, default=250000) # 250000

    config = parser.parse_args()
    print(config)
    main(config)