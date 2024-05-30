from datasets import load_dataset
import wandb
import lightning as L

class Dataset:
    def __init__(self, name: str, split: str):
        self.name = name
        self.split = split
        self.dataset = load_dataset(self.name, split=self.split)

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
    
    
