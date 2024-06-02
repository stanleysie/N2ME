import os
import random
import torch
import numpy as np
import argparse

torch.cuda.empty_cache()

class Config:
    def __init__(self, mode):
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_config()
        self.set_global_seed()

    def init_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--mode', type=str, default=self.mode, help='train or test')

        # Base configurations
        parser.add_argument('--ckpts_dir', type=str, help='models are saved here')
        parser.add_argument('--samples_dir', type=str, default='', help='samples are saved here')
        parser.add_argument('--expr_name', type=str, default='experiment_1', help='name of experiment (will also be the directory where models and samples are stored)')
        parser.add_argument('--model_name', type=str, default='ganimation_modified', help='model to use: ganimation, ganimation_modified')

        parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        parser.add_argument('--img_size', type=int, default=128, help='image dimension')
        parser.add_argument('--c_dim', type=int, default=17, help='number of AUs to use as condition')
        parser.add_argument('--use_18_aus', type=int, default=0, help='use 18 AUs instead of 17 AUs')
        parser.add_argument('--n_critics', type=int, default=5, help='number of D updates per each G update')
        parser.add_argument('--n_workers_train', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--n_workers_test', default=1, type=int, help='# threads for loading data')

        parser.add_argument('--seed', type=int, default=0, help='random seed')
        parser.add_argument('--num_epochs', type=int, default=30, help='total epochs for training D')
        parser.add_argument('--num_epochs_decay', type=int, default=20, help='epochs to start decaying lr')
        parser.add_argument('--load_epoch', type=int, default=-1, help='starting or resuming epoch, set to -1 to automatically start from latest saved model')
        parser.add_argument('--end_epoch', type=int, default=-1, help='end epoch, set to - 1 to automatically end at the latest epoch')
        parser.add_argument('--num_iters_validate', default=1, type=int, help='# batches to use when validating')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--display_freq', type=int, default=500, help='frequency of showing training results on tensorboard')
        parser.add_argument('--save_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--show_params', type=int, default=0, help='show model parameters')

        # Generator configurations    
        parser.add_argument('--G_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
        parser.add_argument('--G_repeat_num', type=int, default=6, help='number of residual blocks in G')

        args = None

        if self.mode == 'train':
            args = self.train_config(parser)
        elif self.mode == 'test':
            args = self.test_config(parser)

        args.ckpts_dir = self.handle_windows_path(args.ckpts_dir)
        args.samples_dir = self.handle_windows_path(args.samples_dir)
        args.use_18_aus = bool(args.use_18_aus)

        self.config = args

        self.init_directories()
        self.save_config(args)
        self.print_config(args)

    def train_config(self, parser):
        # Training configurations
        parser.add_argument('--data_dir', type=str, help='path to data directory')
        parser.add_argument('--imgs_dir', type=str, help='path to images directory')
        parser.add_argument('--aus_file', type=str, help='path to aus file')
        parser.add_argument('--train_ids_file', type=str, default='train_ids.txt', help='path to train_ids.txt')
        parser.add_argument('--test_ids_file', type=str, default='test_ids.txt', help='path to test_ids.txt')

        parser.add_argument('--G_lr', type=float, default=0.0001, help='learning rate of G')
        parser.add_argument('--D_lr', type=float, default=0.0001, help='learning rate for D')
        parser.add_argument('--G_adam_beta1', type=float, default=0.5, help='beta1 for Adam optimizer in G')
        parser.add_argument('--G_adam_beta2', type=float, default=0.999, help='beta2 for Adam optimizer in G')
        parser.add_argument('--D_adam_beta1', type=float, default=0.5, help='beta1 for Adam optimizer in D')
        parser.add_argument('--D_adam_beta2', type=float, default=0.999, help='beta2 for Adam optimizer in D')
        parser.add_argument('--D_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
        parser.add_argument('--D_repeat_num', type=int, default=6, help='number of residual blocks in D')
        parser.add_argument('--lambda_dis', type=float, default=1, help='lambda for real/fake discriminator loss')
        parser.add_argument('--lambda_aus', type=float, default=160, help='lambda for condition discriminator loss')
        parser.add_argument('--lambda_idt', type=float, default=10, help='lambda cycle consistency loss for identity loss')
        parser.add_argument('--lambda_gp', type=float, default=10, help='lambda gradient penalty loss')
        parser.add_argument('--lambda_mask', type=float, default=0.1, help='lambda mask loss')
        parser.add_argument('--lambda_tv', type=float, default=1e-5, help='lambda mask smooth loss')

        parser.add_argument('--train_mode', type=str, default='neutral', help='train mode: neutral, me')
        parser.add_argument('--use_tensorboard', type=int, default=1, help='use tensorboard visualization')
        parser.add_argument('--noise_on_labels', type=int, default=1, help='add noise to labels')
        parser.add_argument('--save_training_images', type=int, default=1, help='save training images')

        args = parser.parse_args()
        args.data_dir = self.handle_windows_path(args.data_dir)
        args.imgs_dir = self.handle_windows_path(args.imgs_dir)
        args.show_params = bool(args.show_params)
        args.use_tensorboard = bool(args.use_tensorboard)
        args.noise_on_labels = bool(args.noise_on_labels)
        args.save_training_images = bool(args.save_training_images)
        
        return args

    def test_config(self, parser):
        # Testing configurations
        parser.add_argument('--test_dir', type=str, help='path to test directory')
        parser.add_argument('--input_dir', type=str, help='path to input images directory')    
        parser.add_argument('--test_mode', type=str, default='all', help='test mode: all, neutral, me')
        parser.add_argument('--keep_size', type=int, default=0, help='keep original image size')

        # Neutral generator configurations
        parser.add_argument('--neutral_model', type=str, default='ganimation', help='neutral model name')
        parser.add_argument('--max_mouth_height', type=float, default=33.015, help='maximum mouth height')
        parser.add_argument('--min_mouth_height', type=float, default=0.0, help='minimum mouth height')
        parser.add_argument('--interpolate', type=int, default=0, help='interpolate between input image to neutral')

        # Micro-expression generator configurations
        parser.add_argument('--samm_aus_dir', type=str, default='', help='path to directory containing aus from SAMM database; empty string means not included in test')
        parser.add_argument('--mmew_aus_dir', type=str, default='', help='path to directory containing aus from MMEW database; empty string means not included in test')
        parser.add_argument('--casme_ii_aus_dir', type=str, default='', help='path to directory containing aus from CASME II database; empty string means not included in test')
        parser.add_argument('--neutral_expr_name', type=str, default='experiment_1', help='name of experiment for neutral generator')
        parser.add_argument('--me_model', type=str, default='ganimation_modified', help='me model name')
        parser.add_argument('--me_summary', type=str, default='me_summary.json', help='micro-expression samples summary')
        parser.add_argument('--me_samples', type=int, default=10, help='number of micro-expression samples per emotion')
        parser.add_argument('--me_emotions', type=str, default='all', help='emotions to use for micro-expression generation: all or comma-separated list of emotions')

        args = parser.parse_args()
        args.test_dir = self.handle_windows_path(args.test_dir)
        args.input_dir = self.handle_windows_path(args.input_dir)
        args.samm_aus_dir = self.handle_windows_path(args.samm_aus_dir)
        args.mmew_aus_dir = self.handle_windows_path(args.mmew_aus_dir)
        args.casme_ii_aus_dir = self.handle_windows_path(args.casme_ii_aus_dir)
        args.show_params = bool(args.show_params)
        args.keep_size = bool(args.keep_size)
        args.interpolate = bool(args.interpolate)
        if args.me_emotions != 'all':
            args.me_emotions = args.me_emotions.split(',')
        
        return args

    def print_config(self, args):
        print(f'\n------------ {args.mode.title()} Configurations -------------')
        for k, v in vars(args).items():
            print(f'{k}: {v}')
        print('-------------- End ----------------\n')

    def save_config(self, args):
        filename = os.path.join(args.ckpts_dir, args.expr_name, f'{args.mode}_config.txt')
        with open(filename, 'w') as f:
            f.write(f'------------ {args.mode.title()} Configurations -------------\n')
            for k, v in vars(args).items():
                f.write('%s: %s\n' % (str(k), str(v)))
            f.write('-------------- End ----------------\n')
    
    def handle_windows_path(self, path):
        drive = path.split(':')[0]
        path = path.replace('\\', '/')
        return path.replace(f'{drive}:/', f'/mnt/{drive.lower()}/')

    def init_directories(self):
        if self.config.mode == 'train':
            # creating directories needed for training
            ckpts_dir = os.path.join(self.config.ckpts_dir, self.config.expr_name)
            if not os.path.exists(ckpts_dir):
                os.makedirs(ckpts_dir)
            if self.config.save_training_images and self.config.samples_dir != '':
                samples_dir = os.path.join(self.config.samples_dir, self.config.expr_name)
                if not os.path.exists(samples_dir):
                    os.makedirs(samples_dir)
    
    def set_global_seed(self):
        seed = self.config.seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)