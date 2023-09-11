import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import math
import datetime
import time
import torchvision.transforms as transforms
"""
experiment:
Traditional training in centralized(E2E) manner
"""
class Branch(nn.Module):
    def __init__(self, branches):
        super(Branch, self).__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
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


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2, stride=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.lrn = nn.LocalResponseNorm(size=3, alpha=5e-05, beta=0.75)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=1)

        self.conv3 = nn.Conv2d(64, 96, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(96, 96, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(96, 64, kernel_size=3, padding=1, stride=1)

        self.fc1 = nn.Linear(1024, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 10)

        self.branch = Branch([
            nn.ReLU(),                              # norm()
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 8 * 8
            nn.LocalResponseNorm(size=3, alpha=5e-05, beta=0.75),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),   # cap(512) # 4 * 4
        ])
        self.branch_fc = nn.Linear(512, 10)

    def froze_S(self):
        S_layers = nn.ModuleList([self.conv3, self.conv4, self.conv5, self.fc1, self.fc2, self.fc3])
        for layer in S_layers:
            for param in layer.parameters():
                param.requires_grad = False

    def unfroze_S(self):
        S_layers = nn.ModuleList([self.conv3, self.conv4, self.conv5, self.fc1, self.fc2, self.fc3])
        for layer in S_layers:
            for param in layer.parameters():
                param.requires_grad = True


    def froze_E_C(self):
        E_C_layers = nn.ModuleList([self.conv1, self.conv2])
        for layer in E_C_layers:
            for param in layer.parameters():
                param.requires_grad = False
        self.branch.froze()


    def unfroze_E_C(self):
        E_C_layers = nn.ModuleList([self.conv1, self.conv2])
        for layer in E_C_layers:
            for param in layer.parameters():
                param.requires_grad = True
        self.branch.unfroze()


    def forward(self, x):
        x = self.conv1(x)       # 32 * 32
        x = self.relu(x)
        x = self.maxpool(x)    # 32 - 3 / 2 + 1 = 16 * 16
        x = self.lrn(x)

        x = self.conv2(x)       # 16
        # branch
        branch = self.branch(x)
        branch = branch.view(branch.size(0), -1)
        branch = self.branch_fc(branch)

        x = self.relu(x)
        x = self.maxpool(x)
        x = self.lrn(x)

        x = self.conv3(x)       # 16 * 16
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.relu(x)

        x = self.maxpool(x)    # 8 * 8 * 64

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)

        return branch, x


train_batch_size = 256
test_batch_size = 128
learning_rate = 0.001
epochs = 10
threshold = 0.4

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 

train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=train_batch_size, shuffle=True,
        num_workers=16, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]),download=True),
    batch_size=test_batch_size, shuffle=False,
    num_workers=16, pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean_criterion = nn.CrossEntropyLoss(reduction="mean").cuda()
sepa_criterion = nn.CrossEntropyLoss(reduction="none").cuda()


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

alpha = 0.5

def train(model, optimizer, Tra_flag):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)  
        optimizer.zero_grad()
        
        branch_output, output = model(data)
        val = get_information(branch_output)   
        
        if Tra_flag:
            loss = (1 - alpha) * mean_criterion(output, target) + alpha * mean_criterion(branch_output, target)  
        else:
            loss = torch.mean((1 - sigmoid(threshold - val)) * sepa_criterion(output, target) + sigmoid(threshold - val) * sepa_criterion(
                branch_output, target))  
        loss.backward()
        optimizer.step()

def get_information(x):
    x = x.to(device)
    x = nn.Softmax(dim=1)(x)
    information = -torch.sum(x * torch.log(x + 1e-8), dim=1)
    return information 


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
            torch.save(model.state_dict(), 'traditional_E2E_highest.pth')
            print("best acc upd: ", best_acc)
            best_acc = correct / total

best_acc = 0


lr_group = [0.001, 0.0005, 0.0001]
cnt = 0
model = AlexNet().to(device)

for i in lr_group:
    learning_rate = i
    optimizer1 = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(20):
        print(f'epochs {cnt} learning rate: {i} ', end="")
        cnt += 1
        train(model, optimizer1, Tra_flag=True)
        test(model)

"""
In traditional group, the below code is not so useful, but to keep the same epochs with Two-stage group, the below code is written here.
There is no doubt you can overlook and delete it.
"""

unknown_num = 20    # not necessarily 20
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for _ in range(unknown_num):
    cnt += 1
    print(f'epochs {cnt} learning rate: {i} ',end="")
    train(model, optimizer, Tra_flag=False) # using L2
    test(model)