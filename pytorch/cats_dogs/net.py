import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleNet, self).__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.b1 = nn.BatchNorm2d(32)
        self.p1 = nn.MaxPool2d(2, 2)
        self.c2 = nn.Conv2d(32, 32, 3, padding=1)
        self.b2 = nn.BatchNorm2d(32)
        self.p2 = nn.MaxPool2d(2, 2)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.b3 = nn.BatchNorm2d(64)
        self.p3 = nn.MaxPool2d(2, 2)
        self.c4 = nn.Conv2d(64, 128, 3, padding=1)
        self.b4 = nn.BatchNorm2d(128)
        self.p4 = nn.MaxPool2d(2, 2)
        self.c5 = nn.Conv2d(128, 256, 3, padding=1)
        self.b5 = nn.BatchNorm2d(256)
        self.p5 = nn.MaxPool2d(2, 2)
        self.l6 = nn.Linear(16 * 256, 128)
        self.l7 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = self.p1(x)
        
        x = F.relu(self.b2(self.c2(x)))
        x = self.p2(x)

        x = F.relu(self.b3(self.c3(x)))
        x = self.p3(x)

        x = F.relu(self.b4(self.c4(x)))
        x = self.p4(x)

        x = F.relu(self.b5(self.c5(x)))
        x = self.p5(x)

        x = x.view(-1, 16 * 256)
        x = F.dropout(F.relu(self.l6(x)))

        return self.l7(x)
