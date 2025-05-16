# coding=utf-8
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import os
import glob
from net import *


# use cuda if available
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

class PolicyBatchDataset(Dataset):
    def __init__(self, data_dir='data/policy_batches', pattern='policyData_part*.pt'):
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
        self.index_map = []

        # 创建索引映射表：每条数据指向其在某个文件的行
        for file_idx, file_path in enumerate(self.file_paths):
            input_tensor, _ = torch.load(file_path)
            for i in range(len(input_tensor)):
                self.index_map.append((file_idx, i))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, row_idx = self.index_map[idx]
        input_tensor, output_tensor = torch.load(self.file_paths[file_idx])
        return input_tensor[row_idx], output_tensor[row_idx]



def trainPolicy(net, output_dir='checkpoints', epoch=10):
    from torch.utils.data import DataLoader, random_split


    os.makedirs(output_dir, exist_ok=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    loss_function = nn.CrossEntropyLoss()

    dataset = PolicyBatchDataset(data_dir='data/policy_batches')
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    batch_size = 100
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    net.to(device)

    best_acc = 0.0
    acc_record = []

    for ep in range(epoch):
        net.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.view(-1).to(device)

            outputs = net(inputs)
            loss = loss_function(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)

            if batch_idx % 10 == 0 and batch_idx != 0:
                avg_loss = total_loss / 10
                accuracy = correct / total * 100
                print(f"Epoch {ep:02d} Batch {batch_idx:04d}  Accuracy: {accuracy:.2f}%  AvgLoss: {avg_loss:.4f}")
                total_loss = 0
                correct = 0
                total = 0

        scheduler.step()

        # ==== Test ====
        net.eval()
        test_loss = 0
        test_correct = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.view(-1).to(device)
                outputs = net(inputs)
                loss = loss_function(outputs, targets)

                test_loss += loss.item()
                test_correct += (outputs.argmax(dim=1) == targets).sum().item()

        test_acc = test_correct / len(test_dataset) * 100
        avg_test_loss = test_loss / len(test_loader)
        current_lr = scheduler.get_last_lr()[0]
        acc_record.append((ep, test_acc, avg_test_loss))

        print(f"Epoch {ep:02d} [Test] Accuracy: {test_acc:.2f}%  AvgLoss: {avg_test_loss:.4f}  LR: {current_lr:.5f}")

        # ==== Save model checkpoint ====
        model_path = os.path.join(output_dir, f'policy_epoch_{ep:02d}.pt')
        torch.save(net.state_dict(), model_path)

        # Keep only the latest 10 models
        all_ckpts = sorted(glob.glob(os.path.join(output_dir, 'policy_epoch_*.pt')))
        if len(all_ckpts) > 10:
            os.remove(all_ckpts[0])

        # Update best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_path = os.path.join(output_dir, 'best_model.pt')
            torch.save(net.state_dict(), best_model_path)
            print(f'New best model saved at epoch {ep:02d} with accuracy {best_acc:.2f}%')

    # Save accuracy log for later analysis
    acc_log_path = os.path.join(output_dir, 'accuracy_log.txt')
    with open(acc_log_path, 'w') as f:
        for ep, acc, loss in acc_record:
            f.write(f'Epoch {ep:02d} | Accuracy: {acc:.2f}% | Loss: {loss:.4f}\n')




if __name__ == '__main__':
    net = PolicyNetwork()
    trainPolicy(net,'checkpoints',30)



