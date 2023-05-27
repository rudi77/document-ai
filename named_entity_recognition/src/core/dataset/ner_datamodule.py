import glob
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader, random_split
from core.dataset.token_classification_dataset import LayoutLMDataset

def custom_collate(batch):
        # batch is a list of sample tuples
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        bbox = torch.stack([item['bbox'] for item in batch])
        return input_ids, attention_mask, labels, bbox


class LayoutLMDataModule(LightningDataModule):
    def __init__(self, csv_dir, image_dir, tokenizer, label2idx, batch_size, num_workers=1):
        super().__init__()
        self.csv_dir = csv_dir
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        full_dataset = LayoutLMDataset(self.csv_dir, 
                                       self.image_dir, 
                                       self.tokenizer, 
                                       self.label2idx)
                                       
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    