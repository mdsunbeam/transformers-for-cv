from typing import Tuple, Optional, Dict, Any

import torch 
from lightning import LightningDataModule 
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST 
from torchvision.transforms import transforms 

class MNISTDataModule(LightningDataModule): 
    """
    LightningDataModule for MNIST dataset. 
    
    Args: 
        data_dir: directory to store the dataset 
        batch_size: batch size 
        num_workers: number of workers 
        val_split: validation split 
        seed: random seed 
    """
    
    def __init__(
            self,
            data_dir: str = "tasks/classification/mnist/data",
            train_val_test_split: Tuple[int, int, int] = (50_000, 5_000, 10_000),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__()
    
        # transforms
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Nomralize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        @property
        def num_classes(self):
            return 10
        
        def prepare_data(self):
            """
            Download the MNIST dataset.
            """
            MNIST(self.data_dir, train=True, download=True)
            MNIST(self.data_dir, train=False, download=True)

        def setup(self, stage: Optional[str] = None):
            """
            Load data and make splits for train, validation, and test.
            """
            if not self.data_train and not self.data_val and not self.data_test:
                trainset = MNIST(self.data_dir, train=True, transform=self.transforms)
                testset = MNIST(self.data_dir, train=False, transform=self.transforms)
                dataset = ConcatDataset([trainset, testset])
                self.data_train, self.data_val, self.data_test = random_split(
                    dataset=dataset,
                    lengths=self.train_val_test_split,
                    generator=torch.Generator().manual_seed(1),
                )

        def train_dataloader(self):
            return DataLoader(
                dataset=self.data_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=True,
            )
        
        def val_dataloader(self):
            return DataLoader(
                dataset=self.data_val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False,
            )
        
        def test_dataloader(self):
            return DataLoader(
                dataset=self.data_test,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False,
            )
        
        def teardown(self, stage: Optional[str] = None):
            """Clean up after fit or test."""
            pass

        def state_dict(self):
            """Miscellaneous items to save to a checkpoint."""
            return {}

        def load_state_dict(self, state_dict: Dict[str, Any]):
            """Items or tasks to do when loading from a checkpoint."""
            pass


if __name__ == "__main__":
    _ = MNISTDataModule()