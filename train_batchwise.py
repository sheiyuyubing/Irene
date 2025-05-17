# coding=utf-8
import torch
import os
import glob
from torch.utils.data import Dataset, DataLoader
from net import PolicyNetwork
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LightDataset(Dataset):
    def __init__(self, file_path):
        self.inputs, self.targets = torch.load(file_path)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# 保存图像并显示
def plot_accuracy_loss(acc_record, save_path='accuracy_plot.png'):
    epochs = [x[0] for x in acc_record]
    accuracies = [x[1] for x in acc_record]
    losses = [x[2] for x in acc_record]

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)', color='tab:blue')
    ax1.plot(epochs, accuracies, label='Accuracy', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='tab:red')
    ax2.plot(epochs, losses, label='Loss', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Test Accuracy and Loss over Epochs')
    fig.tight_layout()
    plt.savefig(save_path)
    print(f'[图已保存] Accuracy & Loss 图像保存到: {save_path}')

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

def trainPolicyBatchwise(net, data_dir='data/policy_batches', output_dir='checkpoints', epoch=10):
    os.makedirs(output_dir, exist_ok=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    net.to(device)
    all_files = sorted(glob.glob(os.path.join(data_dir, 'policy_batch_*.pt')))
    print(f"Found {len(all_files)} batch files.")

    acc_record = []

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
        acc_record.append((ep, ep_acc, ep_loss))

        print(f"Epoch {ep} Summary: Avg Loss: {ep_loss:.4f}, Avg Acc: {ep_acc:.2f}%")

        torch.save(net.state_dict(), os.path.join(output_dir, f'policy_epoch_{ep:02d}.pt'))

    # 保存日志和图像
    acc_log_path = os.path.join(output_dir, 'accuracy_log.txt')
    with open(acc_log_path, 'w') as f:
        for ep, acc, loss in acc_record:
            f.write(f'Epoch {ep:02d} | Accuracy: {acc:.2f}% | Loss: {loss:.4f}\n')

    plot_accuracy_loss(acc_record, os.path.join(output_dir, 'accuracy_plot.png'))

if __name__ == '__main__':
    net = PolicyNetwork()
    trainPolicyBatchwise(net, epoch=30, output_dir='checkpoints')
