import time
import torch
from data.data_loader import CustomDataLoader
from models.model_factory import build_model
from utils.logger import Logger
from config import Config

class Train(Config):
    def __init__(self):
        super().__init__(mode="train")

        self.config.c_dim = self.config.c_dim + 1 if self.config.use_18_aus else self.config.c_dim

        # creating data loader
        self.train_loader = CustomDataLoader(self.config, is_train=True)
        self.test_loader = CustomDataLoader(self.config, is_train=False)

        # loading dataset
        self.train_data = self.train_loader.load_data()
        self.test_data = self.test_loader.load_data()

        self.n_train = len(self.train_loader)
        self.n_test = len(self.test_loader)
        print(f'Train images: {self.n_train}')
        print(f'Test images: {self.n_test}')

        # building models
        self.model = build_model(self.config)
        self.config = self.model.get_config()
        self.device = self.model.get_device()

        # initialize tensorboard logger
        self.logger = Logger(self.config)

        # start training
        self.train()
    

    def train(self):
        self.total_steps = self.config.load_epoch * self.n_train
        self.iters_per_epoch = int(self.n_train / self.config.batch_size)
        end_epoch = self.config.end_epoch if self.config.end_epoch != -1 else self.config.num_epochs

        for epoch in range(self.config.load_epoch+1, end_epoch+1):
            print(f'\n[START - Epoch {epoch} / {self.config.num_epochs}]')
            start_time = time.process_time()

            # train epoch
            self.train_epoch(epoch)

            # update learning rate
            if self.config.num_epochs_decay <= epoch < self.config.num_epochs:
                current_lr = self.model.get_latest_lr()
                self.model.update_lr()
                new_lr = self.model.get_latest_lr()
                self.logger.custom_log(f'\n[UPDATE] G learning rate: {current_lr["G"]} -> {new_lr["G"]}\n[UPDATE] D learning rate: {current_lr["D"]} -> {new_lr["D"]}\n')

            total_epoch_time = int(time.process_time() - start_time)
            print(f'\n[END - Epoch {epoch} / {self.config.num_epochs}] Time Taken: {total_epoch_time} sec ({int(total_epoch_time / 60)} min or {int(total_epoch_time / 3600)} hour)\n')
            self.logger.custom_log(f'\n[END - Epoch {epoch} / {self.config.num_epochs}] Time Taken: {total_epoch_time} sec ({int(total_epoch_time / 60)} min or {int(total_epoch_time / 3600)} hour)')
            
            # save model
            self.model.save(epoch, end=True)
    
        
    def train_epoch(self, epoch):
        self.model.set_train()
        
        for i, data in enumerate(self.train_data):
            # get batch data
            X_real = data['real_img'].to(device=self.device, dtype=torch.float)
            y_real = data['real_aus'].to(device=self.device, dtype=torch.float)
            y_target = data['target_aus'].to(device=self.device, dtype=torch.float)

            # set batch input
            self.model.set_batch_input(X_real, y_real, y_target)

            # optimize model
            self.model.optimize_params(train_G=(i == 0 or (i + 1) % self.config.n_critics == 0))
            
            # update total steps
            self.total_steps += self.config.batch_size

            # print losses
            if i == 0 or (i + 1) % self.config.display_freq == 0:
                self.validate_epoch(epoch, i)
            elif (i + 1) % self.config.print_freq == 0:
                current_losses = self.model.get_latest_losses()
                self.logger.log(losses=current_losses,
                                epoch=epoch,
                                current_iterations=i + 1,
                                epoch_iterations=self.iters_per_epoch,
                                total_steps=self.total_steps,
                                lr=self.model.get_latest_lr(),
                                visualize=False)

            # save model
            if i == 0 or (i + 1) % self.config.save_freq == 0:
                self.model.save(epoch, end=False)


    def validate_epoch(self, epoch, train_idx):
        # display the latest training results
        self.logger.display_images(self.model.get_latest_visuals(), epoch=epoch, total_steps=self.total_steps, is_train=True)
        self.logger.plot_scalars(self.model.get_latest_losses(), total_steps=self.total_steps, is_train=True)

        # set model to eval mode
        self.model.set_eval()

        # evaluate test dataset
        for idx, data in enumerate(self.test_data):
            if idx == self.config.num_iters_validate:
                break

            X_real = data['real_img'].to(device=self.device, dtype=torch.float)
            y_real = data['real_aus'].to(device=self.device, dtype=torch.float)
            if self.config.train_mode == 'neutral':
                y_target = torch.zeros(self.config.batch_size, self.config.c_dim).to(device=self.device, dtype=torch.float)
            else:
                y_target = data['target_aus'].to(device=self.device, dtype=torch.float)

            self.model.set_batch_input(X_real, y_real, y_target)
            self.model.evaluate()
            current_losses = self.model.get_latest_losses()
            self.logger.log(losses=current_losses,
                            epoch=epoch,
                            current_iterations=train_idx + 1,
                            epoch_iterations=self.iters_per_epoch,
                            total_steps=self.total_steps,
                            lr=self.model.get_latest_lr(),
                            visualize=True)
        
        self.logger.display_images(self.model.get_latest_visuals(), epoch=epoch, total_steps=self.total_steps, is_train=False)
        self.logger.plot_scalars(self.model.get_latest_losses(), total_steps=self.total_steps, is_train=False)
        
        # set model to train mode
        self.model.set_train()

if __name__ == "__main__":
    Train()