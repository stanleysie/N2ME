from torch.utils.data import DataLoader
from .dataset import CustomDataset


class CustomDataLoader:
    def __init__(self, config, is_train=True):
        super(CustomDataLoader, self).__init__()
        self._name = 'DataLoader'
        self.config = config
        self.is_train = is_train
        self.n_workers = config.n_workers_train if is_train else config.n_workers_test
        self.create_dataset()

    def create_dataset(self):
        self.dataset = CustomDataset(self.config, self.is_train)
        self.dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size, 
                                    shuffle=False, num_workers=self.n_workers, pin_memory=True)
        print(f'{"Train" if self.is_train else "Test"} dataset was created!')

    def load_data(self):
        return self.dataloader
    
    def __len__(self):
        return len(self.dataset)