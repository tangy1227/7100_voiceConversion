from model_vc import Generator
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn.functional as F
import time
import datetime


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
            code_reconst = self.G(x_identic_psnt, emb_org, None)
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

            # Increment the current iteration
            current_iteration += 1            

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
        

        save_model_path = 'model_checkpoint_{}.ckpt'.format(i + 1)
        torch.save({
            'model': self.G.state_dict(),
            'optimizer': self.g_optimizer.state_dict(),
        }, save_model_path)
        print("Model saved to {} after training.".format(save_model_path))

        self.writer.close()
