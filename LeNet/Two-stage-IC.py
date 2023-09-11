import torch
import torch.nn as nn
import torchvision
import math
import numpy as np
import time
import torch.nn.functional as F
import datetime
"""
experiment:
Two-stage training in distributed(IC-only) manner
"""
class Branch(nn.Module):
    def __init__(self, branches):
        super(Branch, self).__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        # outputs = [branch(x) for branch in self.branches] # here
        for branch in self.branches:
            x = branch(x)
        return x

    def froze(self):
        for branch in self.branches:
            for param in branch.parameters():
                param.requires_grad = False

    def unfroze(self):
        for branch in self.branches:
            for param in branch.parameters():
                param.requires_grad = True

class LeNet(nn.Module):
    def __init__(self, percentTrainKeeps=1):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 5, kernel_size=5, stride=1, padding=3)    # (28 + 6 - 5) / 1 + 1
        self.branch = Branch([  # first branch
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.ReLU(),
            nn.Conv2d(5, 10, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.ReLU()   # 640 * 10
        ])
        self.branch_fc = nn.Linear(640,10)

        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True) # 15
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(5, 10, kernel_size=5, stride=1, padding=3)   # 17
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)     # 9 * 9
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=3)  # 11 * 11
        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)     # 6 * 6
        self.relu4 = nn.ReLU()

        self.fc5 = nn.Linear(720, 84)       # 6 * 6 * 20 == 720
        self.fc6 = nn.Linear(84, 10)        # last branch

        self.percentTrainKeeps = percentTrainKeeps
        self.threshold = nn.Parameter(torch.tensor(0.25))

    def froze_E_C(self):
        for param in self.conv1.parameters():   # E
            param.requires_grad = False
        self.branch.froze()
        for param in self.branch_fc.parameters():
            param.requires_grad = False

    def unfroze_E_C(self):
        for param in self.conv1.parameters():   # E
            param.requires_grad = True
        self.branch.unfroze()
        for param in self.branch_fc.parameters():
            param.requires_grad = True


    def froze_S(self):
        target = [self.pool2, self.pool3, self.pool4,
                  self.relu2, self.relu3, self.relu4,
                  self.conv3, self.conv4, self.fc5, self.fc6]
        for element in target:
            for param in element.parameters():
                param.requires_grad = False

    def unfroze_S(self):
        target = [self.pool2, self.pool3, self.pool4,
                  self.relu2, self.relu3, self.relu4,
                  self.conv3, self.conv4, self.fc5, self.fc6]
        for element in target:
            for param in element.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.conv1(x)

        branch = self.branch(x)
        branch = branch.view(-1, 640)
        branch = self.branch_fc(branch)

        x = self.pool2(x)
        x = self.relu2(x)
        # print(x.shape)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.relu3(x)
        # print(x.shape)

        x = self.conv4(x)
        x = self.pool4(x)
        x = self.relu4(x)
        # print(x.shape)

        x = x.view(-1, 720)
        x = self.fc5(x)
        x = self.fc6(x)

        return branch, x


train_batch_size = 64
test_batch_size = 64
learning_rate = 0.001
epochs = 10
threshold = 0.25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=16)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=16)

mean_criterion = nn.CrossEntropyLoss(reduction="mean").cuda()
sepa_criterion = nn.CrossEntropyLoss(reduction="none").cuda()

alpha = 0.5

def get_information(x):
    x = x.to(device)
    x = nn.Softmax(dim=1)(x)
    information = -torch.sum(x * torch.log(x + 1e-8), dim=1)
    return information


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def train(model, optimizer, Tra_flag):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) 
        optimizer.zero_grad()
        
        branch_output, output = model(data)
        val = get_information(branch_output)    
        
        if Tra_flag:
            loss = (1 - alpha) * mean_criterion(output, target) + alpha * mean_criterion(branch_output, target)  # maybe onehot
        else:
            loss = torch.mean((1 - sigmoid(threshold - val)) * sepa_criterion(output, target) + sigmoid(threshold - val) * sepa_criterion(
                branch_output, target)) 
        loss.backward()
        optimizer.step()
        

def test(model):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        exit_cnt = 0
        for data, target in test_loader:  
            data, target = data.to(device), target.to(device) 
            branch, output = model(data)

            information = get_information(branch)
            _, pred1 = torch.max(branch.data, dim=1)  # Use .values to get the max values
            _, pred2 = torch.max(output.data, dim=1)  # Use .values to get the max values

            below_threshold = information < threshold
            correct_pred1 = (pred1 == target).int()
            correct_pred2 = (pred2 == target).int()

            exit_cnt += torch.sum(below_threshold).item()
            correct += torch.sum(below_threshold * correct_pred1).item() + torch.sum((~below_threshold) * correct_pred2).item()
            total += target.size(0)
        print(f'Exit rate: {exit_cnt / total} Accuracy: {correct / total}')
        global best_acc
        if best_acc < correct / total:
            torch.save(model.state_dict(), "two_stage_E2E_highest.pth")
            best_acc = correct / total
            print("best acc upd: ", best_acc)


lr_group = [0.005, 0.001, 0.0005, 0.0002, 0.0001]   # each learning rate train 5 epochs
best_acc = 0

cnt = 0

model = LeNet().to(device)
for i in lr_group:
    learning_rate = i
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(5):
        cnt += 1
        print(f'epochs {cnt} learning rate: {i} ',end="")
        model.froze_S()
        train(model, optimizer, Tra_flag=True)  # using L1
        model.unfroze_S()

        model.froze_E_C()   
        train(model, optimizer, Tra_flag=True)  # using L1
        model.unfroze_E_C()
        test(model)

# stage 2
unknown_num = 20    # not necessarily 20
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for _ in range(unknown_num):
    cnt += 1
    print(f'epochs {cnt} learning rate: {i} ',end="")
    model.froze_S()
    train(model, optimizer, Tra_flag=False) # using L2
    model.unfroze_S()

    model.froze_E_C()
    train(model, optimizer, Tra_flag=False) # using L2
    model.unfroze_E_C()
    test(model)