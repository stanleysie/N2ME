import torch
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import grad, Variable
from torch.optim.lr_scheduler import PolynomialLR
from models.base_model import BaseModel
from models.util import *
from models.ganimation.networks import Generator, Discriminator
from utils.util import tensor2im, tensor2maskim, plot_au

def build_model(config):
    model = Model(config)
    return model

class Model(BaseModel):
    ###############################################################
    ############################ MODEL ############################
    ###############################################################
    def __init__(self, config):
        super(Model, self).__init__(config)
        self._name = 'GANimation'

        # building models
        self.build_models(show_params=config.show_params)

        # restore models if available
        self.restore_models()

        if self.config.mode == 'train':
            # initialize scheduler
            self.initialize_scheduler()

        # initialize losses
        self.initialize_losses()

    def build_models(self, show_params=False):
        print('\nCreating models...')
        # initialize networks
        self.G = Generator(conv_dim=self.config.G_conv_dim, 
                            c_dim=self.config.c_dim, 
                            repeat_num=self.config.G_repeat_num).to(self.device)
        if show_params:
            self.print_network(self.G)

        if self.config.mode == 'train':
            self.D = Discriminator(img_size=self.config.img_size,
                                    conv_dim=self.config.D_conv_dim, 
                                    c_dim=self.config.c_dim, 
                                    repeat_num=self.config.D_repeat_num).to(self.device)
            if show_params:
                self.print_network(self.D)

            # initialize learning rate
            self.G_lr = self.config.G_lr
            self.D_lr = self.config.D_lr

            # initialize optimizers
            self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.G_lr, betas=(self.config.G_adam_beta1, self.config.G_adam_beta2))
            self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.D_lr, betas=(self.config.D_adam_beta1, self.config.D_adam_beta2))
    
    def initialize_scheduler(self):
        # initialize scheduler
        total_epochs_decay = self.config.num_epochs - self.config.num_epochs_decay
        self.G_scheduler = PolynomialLR(self.G_optimizer, total_iters=total_epochs_decay, power=1.0)
        self.D_scheduler = PolynomialLR(self.D_optimizer, total_iters=total_epochs_decay, power=1.0)

    def initialize_losses(self):
        self.criterionGAN = GANLoss
        self.criterionL1 = nn.L1Loss()
        self.criterionMSE = nn.MSELoss()
        self.criterionTV = TVLoss

        self.G_losses = ['G_loss_fake', 'G_loss_aus', 'G_loss_idt', 'G_loss_fake_mask', 'G_loss_rec_mask', 'G_loss_fake_mask_smooth', 'G_loss_rec_mask_smooth']
        self.D_losses = ['D_loss_real', 'D_loss_fake', 'D_loss_aus', 'D_loss_gp']

    def forward(self):
        # generate attention mask and color mask
        self.A_mask, self.C_mask = self.G.forward(self.X_real, self.y_target)
        self.X_fake = (1 - self.A_mask) * self.C_mask + self.A_mask * self.X_real

        if self.config.mode == 'train':
            # reconstruct real images
            self.A_mask_rec, self.C_mask_rec = self.G.forward(self.X_fake, self.y_real)
            self.X_rec = (1 - self.A_mask_rec) * self.C_mask_rec + self.A_mask_rec * self.X_fake
    
    def backward_G(self, is_train=True):
        # D(X_fake)
        pred_fake, y_target_fake = self.D.forward(self.X_fake)
        self.G_loss_fake = self.criterionGAN(pred_fake, True)
        self.G_loss_aus = self.criterionMSE(y_target_fake, self.y_target)

        # identity loss
        self.G_loss_idt = self.criterionL1(self.X_rec, self.X_real)

        # attention mask loss
        self.G_loss_fake_mask = torch.mean(self.A_mask)
        self.G_loss_rec_mask = torch.mean(self.A_mask_rec)
        self.G_loss_fake_mask_smooth = self.criterionTV(self.A_mask)
        self.G_loss_rec_mask_smooth = self.criterionTV(self.A_mask_rec)

        # combine losses
        self.G_loss_total = self.config.lambda_dis * self.G_loss_fake + \
                            self.config.lambda_aus * self.G_loss_aus + \
                            self.config.lambda_idt * self.G_loss_idt + \
                            self.config.lambda_mask * (self.G_loss_fake_mask + self.G_loss_rec_mask) + \
                            self.config.lambda_tv * (self.G_loss_fake_mask_smooth + self.G_loss_rec_mask_smooth)
        
        if is_train:
            self.G_loss_total.backward()

    def backward_D(self, is_train=True):
        # D(X_real)
        pred_real, y_target_fake = self.D.forward(self.X_real)
        self.D_loss_real = self.criterionGAN(pred_real, True)
        self.D_loss_aus = self.criterionMSE(y_target_fake, self.y_real)

        # D(X_fake)
        pred_fake, _ = self.D.forward(self.X_fake.detach())
        self.D_loss_fake = self.criterionGAN(pred_fake, False)

        # combine losses
        self.D_loss_total = self.config.lambda_dis * (self.D_loss_real + self.D_loss_fake) + \
                            self.config.lambda_aus * self.D_loss_aus
            
        if is_train:
            # gradient penalty
            self.D_loss_gp = self.gradient_penalty_D(self.X_real, self.X_fake)
            self.D_loss_total += self.config.lambda_gp * self.D_loss_gp
            self.D_loss_total.backward()
    
    def gradient_penalty_D(self, X_real, X_fake):
        alpha = torch.rand(X_real.size(0), 1, 1, 1).to(self.device).expand_as(X_real)
        interpolated = Variable((1 - alpha) * X_fake.data + alpha * X_real.data, requires_grad=True)
        interpolated_prob, _ = self.D.forward(interpolated)

        # compute gradients
        gradients = grad(outputs=interpolated_prob, 
                        inputs=interpolated,
                        grad_outputs=torch.ones(interpolated_prob.size()).to(self.device),
                        retain_graph=True,
                        create_graph=True,
                        only_inputs=True)[0]
        
        # penalize gradients
        gradients = gradients.view(gradients.size(0), -1)
        grad_l2_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1))
        return torch.mean((grad_l2_norm - 1) ** 2)
    
    def optimize_params(self, train_G):
        # forward pass
        self.forward()

        # update discrminator
        self.set_requires_grad(self.D, True)
        self.D_optimizer.zero_grad()
        self.backward_D()
        self.D_optimizer.step()

        if train_G:
            # update generator
            self.set_requires_grad(self.D, False)
            self.G_optimizer.zero_grad()
            self.backward_G()
            self.G_optimizer.step()

    def evaluate(self):
        with torch.no_grad():
            self.forward()
            if self.config.mode == 'train':
                self.backward_G(is_train=False)
                self.backward_D(is_train=False)
    
    def inference(self):
        self.evaluate()
        return tensor2im(self.X_fake.data)

    def update_lr(self):
        # updating G
        old_lr = self.G_scheduler.get_last_lr()
        self.G_scheduler.step()
        new_lr = self.G_scheduler.get_last_lr()
        print(f'\n[UPDATE] G learning rate: {old_lr} -> {new_lr}')
        
        # updating D
        old_lr = self.D_scheduler.get_last_lr()
        self.D_scheduler.step()
        new_lr = self.D_scheduler.get_last_lr()
        print(f'\n[UPDATE] D learning rate: {old_lr} -> {new_lr}')
    
    def get_latest_lr(self):
        G_lr = self.G_scheduler.get_last_lr()[0]
        D_lr = self.D_scheduler.get_last_lr()[0]

        return OrderedDict([('G', G_lr), ('D', D_lr)])
    
    def get_latest_losses(self, model=None):
        losses = []
        if model == 'G':
            losses += self.G_losses
        elif model == 'D':
            losses += self.D_losses
        else:
            losses += self.G_losses + self.D_losses

        loss_dict = OrderedDict()

        for loss in losses:
            if hasattr(self, loss):
                loss_dict[loss] = getattr(self, loss).item()

        return loss_dict
    
    def get_latest_visuals(self):
        y_real = self.y_real.cpu()[0, ...].numpy()
        y_target = self.y_target.cpu()[0, ...].numpy()

        visuals = OrderedDict([
            ('X_real', tensor2im(self.X_real)),
            ('X_fake', tensor2im(self.X_fake.data)),
            ('X_rec', tensor2im(self.X_rec.data)),
            ('A_mask', tensor2maskim(self.A_mask.data)),
            ('C_mask', tensor2im(self.C_mask.data)),
            ('batch_X_real', tensor2im(self.X_real, idx=-1)),
            ('batch_A_mask', tensor2maskim(self.A_mask.data, idx=-1, nrows=5)),
            ('batch_C_mask', tensor2im(self.C_mask.data, idx=-1)),
            ('batch_X_fake', tensor2im(self.X_fake.data, idx=-1)),
            ('batch_X_rec', tensor2im(self.X_rec.data, idx=-1))
        ])

        # plot aus
        visuals['X_real'] = plot_au(visuals['X_real'], y_real)
        visuals['X_fake'] = plot_au(visuals['X_fake'], y_target)
        visuals['X_rec'] = plot_au(visuals['X_rec'], y_real)

        return visuals
    
    def set_batch_input(self, X_real, y_real, y_target):
        self.X_real = X_real
        self.y_real = y_real
        self.y_target = y_target
    
    def set_train(self):
        self.G.train()
        self.D.train()

    def set_eval(self):
        self.G.eval()