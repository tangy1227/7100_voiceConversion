from model_vc import Generator
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn.functional as F
import time
import datetime
import numpy as np
import pickle
import os
import math
import librosa


class Solver(object):

    def __init__(self, vcc_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.log_step = config.log_step
        self.val_step = 1000

        # Build the model and tensorboard.
        self.build_model()

        # tensorboard
        self.writer = SummaryWriter(log_dir=config.log_dir)
        self.num_ckpt = config.num_ckpt

            
    def build_model(self):
        self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), 0.0001)
        self.G.to(self.device)
        

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()

    def pad_seq(self, x, base=32):
        len_out = int(base * math.ceil(float(x.shape[0])/base))
        len_pad = len_out - x.shape[0]
        assert len_pad >= 0
        return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad
    
    def log_spec_dB_dist(self, x, y):
        log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
        diff = x - y
        return log_spec_dB_const * math.sqrt(np.inner(diff, diff))    

    def validate(self, current_iteration):
        """
        Need to have similar format with meta4test
        [filename, speaker embedding, spectrogram]
        """

        path = '/home/ytang363/7100_voiceConversion/VCTK-Corpus-0.92/spmel-16k'
        train_meta_path = os.path.join(path, "train_xvec.pkl")
        train_meta = pickle.load(open(train_meta_path, "rb"))
        random_indices = np.random.choice(len(train_meta), 1, replace=False) # pick 5

        for ind in random_indices:
            spk_meta = train_meta[ind]

            trg_ind = np.random.randint(0, len(train_meta)) # find a new ind for target
            trg_spk_meta = train_meta[trg_ind]

            org_uttr_list = spk_meta[2:]
            trg_uttr_list = trg_spk_meta[2:]

            test_uttr = ''    
            org_uttr_name = [file_name.split('_')[1] for file_name in org_uttr_list]
            trg_uttr_name = [file_name.split('_')[1] for file_name in trg_uttr_list]
            while test_uttr not in trg_uttr_name:
                uttr_ind = np.random.randint(0, len(org_uttr_name))
                test_uttr = org_uttr_name[uttr_ind]
            trg_uttr_ind = trg_uttr_name.index(org_uttr_name[uttr_ind])

            org_spk_emb, trg_spk_emb = spk_meta[1], trg_spk_meta[1]
            org_uttr = np.load(os.path.join(path, org_uttr_list[uttr_ind]))
            trg_uttr = np.load(os.path.join(path, trg_uttr_list[trg_uttr_ind]))

            org_uttr, len_pad = self.pad_seq(org_uttr)
            
            with torch.no_grad():
                x_real = torch.from_numpy(org_uttr[np.newaxis, :, :]).to(self.device) # uttr mel spec: [2, 128, 80]
                emb_org = torch.from_numpy(org_spk_emb[np.newaxis, :]).to(self.device) # [2, 512]
                emb_trg = torch.from_numpy(trg_spk_emb[np.newaxis, :]).to(self.device)

                # Forward pass
                _, x_identic_psnt, _ = self.G(x_real, emb_org, emb_trg)
                if len_pad == 0:
                    uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
                else: # excluding a padding portion
                    uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
                
                # dtw mcd
                cost_function = self.log_spec_dB_dist
                print(type(trg_uttr))
                print(type(uttr_trg))
                min_cost, _ = librosa.sequence.dtw(trg_uttr, uttr_trg, metric=cost_function)                      

                value = None

        # # Calculate average validation loss

        # # Log the validation loss to TensorBoard
        # for tag, value in val_loss.items():
        #     self.writer.add_scalar(tag + '_val', value, current_iteration)

        # Set the model back to training mode
        self.G.train()
    

    #=====================================================================================================================================#
              
                
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader
        
        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd']
            
        # Start training.
        print('Start training...')
        start_time = time.time()

        # Create a variable to keep track of the current iteration
        current_iteration = 0
        for i in range(self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            try:
                x_real, emb_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, emb_org = next(data_iter)

            x_real = x_real.to(self.device) 
            emb_org = emb_org.to(self.device) 


            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #
            
            self.G = self.G.train()
                        
            # Identity mapping loss
            # x_real shape: torch.Size([2, 128, 80]), emb_org ([2, 256])
            x_identic, x_identic_psnt, code_real = self.G(x_real, emb_org, emb_org)
            x_identic = x_identic.squeeze(1)
            x_identic_psnt = x_identic_psnt.squeeze(1)            
            # print(f'x_real: {x_real.shape}, x_identic: {x_identic.shape}, x_identic_psnt: {x_identic_psnt.shape}')
            g_loss_id = F.mse_loss(x_real, x_identic)   
            g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt)   
            
            # Code semantic loss.
            code_reconst = self.G(x_identic_psnt, emb_org, None) # [2, 256]
            # print(f'code_real: {code_real.shape}, code_reconst: {code_reconst.shape}')
            g_loss_cd = F.l1_loss(code_real, code_reconst)

            # Backward and optimize.
            g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss = {}
            loss['G/loss_id'] = g_loss_id.item()
            loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
            loss['G/loss_cd'] = g_loss_cd.item()

            # Log the loss values to TensorBoard
            for tag, value in loss.items():
                self.writer.add_scalar(tag, value, current_iteration)        

            # Save at each num_ckpt
            if (i + 1) % self.num_ckpt == 0:
                checkpoint_path = 'model_checkpoint_{}.ckpt'.format(i + 1)
                torch.save({
                            'model': self.G.state_dict(),
                            'optimizer': self.g_optimizer.state_dict(),
                            }, checkpoint_path)
                print("Model checkpoint saved at iteration {}.".format(i + 1))

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)
            
            # Validation
            # if (i+1) % self.log_step == 0:
            print('Start validation...')
            self.validate(current_iteration)

            # Increment the current iteration
            current_iteration += 1                    

        save_model_path = 'model_checkpoint_{}.ckpt'.format(i + 1)
        torch.save({
            'model': self.G.state_dict(),
            'optimizer': self.g_optimizer.state_dict(),
        }, save_model_path)
        print("Model saved to {} after training.".format(save_model_path))

        self.writer.close()
