import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as pyplot
import math
import time

from torch.autograd import Variable

"""
experiment:
Traditional training in distributed(IC-only) manner
"""

__all__ = ['ResNet', 'resnet20']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.branch_fc = nn.Linear(64, num_classes)

        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  # torch.Size([64, 16, 32, 32])

        branch = F.avg_pool2d(out, (int(out.size()[3] / 2), int(out.size()[3] / 2)))   # torch.Size([64, 16, 2, 2])
        branch = branch.view(branch.size(0), -1)
        branch = self.branch_fc(branch)

        out = self.layer2(out)  # torch.Size([64, 32, 16, 16])
        out = self.layer3(out)  # torch.Size([64, 64, 8, 8])
        out = F.avg_pool2d(out, out.size()[3])  # torch.Size([64, 64, 1, 1])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return branch, out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


alpha = 0.5
train_batch_size = 256
test_batch_size = 128
learning_rate = 0.1
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

epochs = 120

criterion = nn.CrossEntropyLoss().cuda()


def get_information(x):
    x = x.to(device)
    x = nn.Softmax(dim=1)(x)
    information = -torch.sum(x * torch.log(x + 1e-8), dim=1)
    return information 

mean_criterion = nn.CrossEntropyLoss(reduction="mean").cuda()
sepa_criterion = nn.CrossEntropyLoss(reduction="none").cuda()

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def train(model, optimizer, Tra_flag=True):
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

ccnt = 0

def test(model,fd):
    model.eval()
    global ccnt
    ccnt += 1
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
            torch.save(model.state_dict(), "traditional_IC_highest.pth")
            best_acc = correct / total
            print("best acc upd: ", best_acc)


best_prec1 = 0
cnt = 0

model = resnet20().to(device)
optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70, 100], last_epoch=-1) # when reach milestone num,lr = lr * 0.1

for epoch in range(1, epochs + 1): 
    cnt += 1
    print('epochs: ', cnt, end=" ")
    print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']), end=" ")
    model.froze_S()
    train(model, optimizer, Tra_flag=True)
    model.unfroze_S()

    model.froze_E_C()
    train(model, optimizer, Tra_flag=True)
    model.unfroze_E_C()
    lr_scheduler.step()
    test(model)

"""
In traditional group, the below code is not so useful, but to keep the same epochs with Two-stage group, the below code is written here.
There is no doubt you can overlook and delete it.
"""

unknown_num = 20    # not necessarily 20
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for _ in range(unknown_num):
    cnt += 1
    print(f'epochs {cnt} learning rate: {learning_rate} ',end="")
    model.froze_S()
    train(model, optimizer, Tra_flag=True)
    model.unfroze_S()

    model.froze_E_C()
    train(model, optimizer, Tra_flag=True)
    model.unfroze_E_C()
    test(model)