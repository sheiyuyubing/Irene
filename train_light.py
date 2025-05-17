# coding=utf-8
from torch.utils.data import Dataset, DataLoader
import torch
import os

class LightDataset(Dataset):
    def __init__(self, file_path):
        self.input_tensor, self.output_tensor = torch.load(file_path)

    def __len__(self):
        return len(self.input_tensor)

    def __getitem__(self, idx):
        return self.input_tensor[idx], self.output_tensor[idx]

if __name__ == '__main__':
    device = torch.device('cpu')
    dataset = LightDataset('data/policy_batches/policy_batch_029.pt')
    print(f"Loaded {len(dataset)} samples")

    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    for batch_idx, (inputs, targets) in enumerate(loader):
        print(f"Batch {batch_idx}: {inputs.shape}, {targets.shape}")
        if batch_idx >= 2:  # 只跑几个 batch 看是否正常
            break
