import torch
import os
import glob
from torch.utils.data import Dataset, DataLoader
from net import PolicyNetwork
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LightDataset(Dataset):
    def __init__(self, file_path):
        self.inputs, self.targets = torch.load(file_path)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def train_on_file(net, file_path, optimizer, loss_fn, batch_size=64):
    dataset = LightDataset(file_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    net.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.view(-1).to(device)

        outputs = net(inputs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / len(loader)
    acc = correct / total * 100
    return avg_loss, acc

def trainPolicyBatchwise(net, data_dir='data/policy_batches', output_dir='checkpoints', epoch=5):
    os.makedirs(output_dir, exist_ok=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    net.to(device)
    all_files = sorted(glob.glob(os.path.join(data_dir, 'policy_batch_*.pt')))
    print(f"Found {len(all_files)} batch files.")

    for ep in range(epoch):
        print(f"\n=== Epoch {ep} ===")
        total_epoch_loss = 0
        total_epoch_acc = 0

        for file in all_files:
            avg_loss, acc = train_on_file(net, file, optimizer, loss_fn)
            total_epoch_loss += avg_loss
            total_epoch_acc += acc
            print(f"[{os.path.basename(file)}] Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")

        scheduler.step()
        ep_loss = total_epoch_loss / len(all_files)
        ep_acc = total_epoch_acc / len(all_files)
        print(f"Epoch {ep} Summary: Avg Loss: {ep_loss:.4f}, Avg Acc: {ep_acc:.2f}%")

        torch.save(net.state_dict(), os.path.join(output_dir, f'policy_epoch_{ep:02d}.pt'))

