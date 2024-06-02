import os
import glob
import torch


class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = os.path.join(config.ckpts_dir, config.expr_name)


    ########################### MODELS ############################
    def restore_models(self):
        print('\nChecking saved models...')
        epoch = self.check_epoch(self.config.load_epoch)
        if epoch > 0:
            self.load_network(self.G, 'G', epoch)
            if self.config.mode == 'train':
                self.load_network(self.D, 'D', epoch)
                self.load_optimizer(self.G_optimizer, 'G', epoch)
                self.load_optimizer(self.D_optimizer, 'D', epoch)

                for param_group in self.G_optimizer.param_groups:
                    self.G_lr = param_group['lr']
                for param_group in self.D_optimizer.param_groups:
                    self.D_lr = param_group['lr']
                
    def save(self, epoch, end):
        if end:
            print(f'\nSaving final models for epoch {epoch}...')
        else:
            print('\nSaving models...')
        self.save_network(self.G, 'G', epoch)
        self.save_network(self.D, 'D', epoch)
        self.save_optimizer(self.G_optimizer, 'G', epoch)
        self.save_optimizer(self.D_optimizer, 'D', epoch)
        print()


    ################### NETWORKS AND OPTIMIZERS ###################
    def set_requires_grad(self, parameters, requires_grad=False):
        if not isinstance(parameters, list):
            parameters = [parameters]
        for param in parameters:
            if param is not None:
                param.requires_grad = requires_grad

    def save_optimizer(self, optimizer, label, epoch):
        filename = f'net_epoch_{epoch}_{label}_optim.pth'
        path = os.path.join(self.save_dir, filename)
        torch.save(optimizer.state_dict(), path)
        print(f'Optimizer saved: {path}')
    
    def load_optimizer(self, optimizer, label, epoch):
        filename = f'net_epoch_{epoch}_{label}_optim.pth'
        path = os.path.join(self.save_dir, filename)
        assert os.path.exists(path), 'Optimizer file not found.'

        optimizer.load_state_dict(torch.load(path))
        print(f'Optimizer loaded: {path}')

    def save_network(self, network, label, epoch):
        filename = f'net_epoch_{epoch}_{label}.pth'
        path = os.path.join(self.save_dir, filename)
        torch.save(network.state_dict(), path)
        print(f'Network saved: {path}')
    
    def load_network(self, network, label, epoch):
        filename = f'net_epoch_{epoch}_{label}.pth'
        path = os.path.join(self.save_dir, filename)
        assert os.path.exists(path), 'Network file not found.'

        network.load_state_dict(torch.load(path))
        print(f'Network loaded: {path}')
    
    def print_network(self, network):
        n_params = 0
        for param in network.parameters():
            n_params += param.numel()
        
        print(network)
        print(f'Total parameters: {n_params}')
    
    def check_epoch(self, epoch):
        expr_dir = os.path.join(self.config.ckpts_dir, self.config.expr_name)

        load_epoch = 0
        if os.path.exists(f'{expr_dir}/net_epoch_{epoch}_G.pth') \
            and os.path.exists(f'{expr_dir}/net_epoch_{epoch}_D.pth') \
            and os.path.exists(f'{expr_dir}/net_epoch_{epoch}_G_optim.pth') \
            and os.path.exists(f'{expr_dir}/net_epoch_{epoch}_D_optim.pth'):
            print(f'Model found!')
            load_epoch = epoch
        elif epoch == -1 or epoch > 0:
            if epoch == -1:
                print(f'Using the latest saved models.')
            
            # find the latest epochs
            files = glob.glob(f'{expr_dir}/net_epoch_*.pth')
            files = [file.split('/')[-1] for file in files]

            if epoch > 0:
                epochs = [int(file.split('_')[2]) for file in files if int(file.split('_')[2]) == epoch]
                if len(epochs) == 0:
                    epochs = [int(file.split('_')[2]) for file in files]
                    load_epoch = max(epochs)
                    print(f'No model found. Using the latest saved model.')
                else:
                    load_epoch = max(epochs)
                    print(f'Model found!')
            else:
                if len(files) == 0:
                    print(f'No model found. Starting from epoch 1.')
                else:
                    epochs = [int(file.split('_')[2]) for file in files]
                    load_epoch = max(epochs)
                    print(f'Model found!')
        else:
            print(f'No saved models found. Starting from epoch 1.')
        
        # updating the load epoch
        self.config.load_epoch = load_epoch
        return load_epoch
    

    ############################ UTILS ############################  
    def get_config(self):
        return self.config
    
    def get_device(self):
        return self.device