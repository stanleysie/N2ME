import os
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, config):
        self.config = config
        self.log_path = os.path.join(config.ckpts_dir, config.expr_name, 'logs.txt')
        self.writer = None
        if config.use_tensorboard:
            self.writer = SummaryWriter(os.path.join(config.ckpts_dir, config.expr_name))

        with open(self.log_path, 'a') as f:
            f.write(f'\n========== TRAINING LOSS ({time.strftime("%b %d, %Y - %H:%M:%S")}) ==========\n')
    
    def __del__(self):
        if self.writer:
            self.writer.close()
    
    def log(self, losses, epoch, current_iterations, epoch_iterations, total_steps, lr, visualize):
        with open(self.log_path, 'a') as f:
            log_time = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f'[{log_time}] Epoch: {epoch}, Steps: {total_steps}, Iterations: {current_iterations}/{epoch_iterations} {"[VISUALIZE]" if visualize else ""}')
            message = ''
            for loss_name, loss_value in losses.items():
                message += f'{loss_name}: {loss_value:.4f}, '
            message += f'G_lr: {lr["G"]}, D_lr: {lr["D"]}'
            print(message)

        with open(self.log_path, 'a') as f:
            f.write(f'[{log_time}] Epoch: {epoch}, Iterations: {current_iterations}/{epoch_iterations}\n')
            f.write(f'{message}\n')
    
    def custom_log(self, message):
        with open(self.log_path, 'a') as f:
            f.write(f'{message}\n')
    
    def plot_scalars(self, scalars, total_steps, is_train):
        if self.writer:
            for label, scalar in scalars.items():
                name = f'{"train" if is_train else "test"}/{label}'
                self.writer.add_scalar(name, scalar, total_steps)

    def display_images(self, visuals, epoch, total_steps, is_train):
        if self.config.save_training_images:
            row, col = 2, visuals.__len__() // 2
            fig, ax = plt.subplots(row, col, figsize=(10, 5), dpi=180)
            keys = list(visuals.keys())
            
            k = 0
            for i in range(row):
                for j in range(col):
                    label = keys[k]
                    img_numpy = visuals[label]
                    name = f'{"train" if is_train else "test"}/{label}'
                    ax[i][j].imshow(img_numpy)
                    ax[i][j].axis('off')
                    ax[i][j].set_title(name, fontsize=10)
                    k += 1
            
            plt.savefig(f"{os.path.join(self.config.samples_dir, self.config.expr_name)}/steps_{epoch:02d}_{total_steps:09d}_{'train' if is_train else 'test'}")
            plt.close(fig)
            
#         if self.writer:
#             for label, img_numpy in visuals.items():
#                 name = f'{"train" if is_train else "test"}/{label}'
#                 self.writer.add_image(name, img_numpy, total_steps, dataformats='HWC')