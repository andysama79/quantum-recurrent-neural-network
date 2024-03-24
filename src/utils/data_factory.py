import torch
from torch.utils.data import Dataset, DataLoader

class QuantumDataset(Dataset):
    def __init_(self, sentences, targets):
        self.sentences = sentences
        self.targets = targets

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sentences[idx]), torch.tensor(self.targets[idx])
    
class DataFactory:
    def __init__(self, shard, num_shards, batch_size, seq_len, sentences, targets):
        self.shard = shard
        self.num_shards = num_shards
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.sentences = sentences
        self.targets = targets
        self.dataset = QuantumDataset(self.sentences, self.targets)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def next_batch(self):
        for idx, batch in enumerate(self.dataloader):
            if idx % self.num_shards == self.shard:
                yield batch

