import os
import pickle
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
from utils import opencv

class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()
        self._name = 'BaseDataset'

        self.IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        ]

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self.root

    def create_transform(self):
        self.transform = transforms.Compose([])

    def get_transform(self):
        return self.transform

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self.IMG_EXTENSIONS)

    def is_csv_file(self, filename):
        return filename.endswith('.csv')


class CustomDataset(BaseDataset):
    def __init__(self, config, is_train=True):
        super(Dataset, self).__init__()
        self._name = 'Dataset'
        self.config = config
        self.is_train = is_train
        self.n_workers = config.n_workers_train if is_train else config.n_workers_test
        self.create_transform()
        self.read_dataset()
    
    def read_dataset(self):
        # setting up paths to dataset
        self.root = self.config.data_dir
        self.imgs_dir = os.path.join(self.root, self.config.imgs_dir)

        # get images ids
        ids_filepath = self.config.train_ids_file if self.is_train else self.config.test_ids_file
        ids_filepath = os.path.join(self.root, ids_filepath)
        self.ids = self.read_ids(ids_filepath)

        # get aus
        aus_filepath = os.path.join(self.root, self.config.aus_file)
        self.aus = self.read_pkl(aus_filepath)

        # get either ids for training or testing
        self.ids = list(set(self.ids).intersection(set(self.aus.keys())))

        # dataset size
        self.dataset_size = len(self.ids)
    
    def create_transform(self):
        transform_list = [transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                   std=[0.5, 0.5, 0.5])]
                                                   
        self.transform = transforms.Compose(transform_list)

    def read_ids(self, filepath):
        with open(filepath, 'r') as f:
            ids = f.readlines()
        return [id.split('.')[0] for id in ids]
    
    def read_pkl(self, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f, encoding='latin1')

    def get_img_by_id(self, id):
        filename = os.path.join(self.imgs_dir, f'{id}.jpg')
        return opencv.read_img(filename)
    
    def get_aus_by_id(self, id):
        if id in self.aus.keys():
            # normalize the values of aus into between 0 and 1
            return self.aus[id]/5.0
        return None

    def generate_random_aus(self, noise=False):
        aus = None
        while aus is None:
            rand_sample_id = self.ids[random.randint(0, self.dataset_size-1)]
            aus = self.get_aus_by_id(rand_sample_id)
            if noise:
                aus += np.random.uniform(-0.1, 0.1, self.config.c_dim)
                aus[aus < 0] = 0
        return aus

    def __getitem__(self, index):
        assert (index < self.dataset_size)

        real_img = None
        real_aus = None
        while real_img is None or real_aus is None:
            # get sample data
            sample_id = self.ids[index]

            real_img = self.get_img_by_id(sample_id)
            real_aus = self.get_aus_by_id(sample_id)

            if real_img is None:
                print(f'Error in reading image {sample_id}.jpg, skipping sample.')
            if real_aus is None:
                print(f'Error in reading aus for {sample_id}.jpg, skipping sample.')

        target_aus = self.generate_random_aus(noise=self.config.noise_on_labels)

        # transform data
        img = self.transform(Image.fromarray(real_img))

        # pack data
        sample = {
            'real_img': img,
            'real_aus': real_aus,
            'target_aus': target_aus
        }

        return sample
    
    def __len__(self):
        return self.dataset_size