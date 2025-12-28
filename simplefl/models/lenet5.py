from .model_base import *
from .model_fn import *


class LeNet5(nn.Module, Model_fn):
    def __init__(self, args, num_classes=10):

        super(LeNet5, self).__init__()

        self.args = args
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 10, 3)
        self.fc1 = nn.Linear(10 * 5 * 5, 100)
        # self.fc2 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(100, num_classes)
        self.dropout = nn.Dropout()
        Model_fn.__init__(self, args, loss_fn=torch.nn.CrossEntropyLoss())
        # Model_fn.__init__(self, args, loss_fn=nl)

    def forward(self, x):
        x = x['pixels']
        x = torch.max_pool2d(torch.relu(self.conv1(x)), 2)
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # x = torch.softmax(x)
        return x


# class LeNet5(torch.nn.Module, Model_fn):

#     def __init__(self, args, num_classes=10):
#         super(LeNet5, self).__init__()
#         self.args = args
#         self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
#                                          torch.nn.ReLU(),
#                                          torch.nn.Conv2d(
#                                              8, 16, kernel_size=3, stride=1, padding=1),
#                                          torch.nn.ReLU(),
#                                          torch.nn.MaxPool2d(stride=2, kernel_size=2))
#         self.dense = torch.nn.Sequential(torch.nn.Linear(14*14*16, 128),
#                                          torch.nn.ReLU(),
#                                          torch.nn.Dropout(p=0.5),
#                                          torch.nn.Linear(128, num_classes))
#         Model_fn.__init__(self, args, loss_fn=torch.nn.CrossEntropyLoss())

#     def forward(self, x):
#         x = x['pixels']
#         x = self.conv1(x)
#         x = x.flatten(1)
#         x = self.dense(x)
#         return x

class Dense(nn.Module, Model_fn):
    def __init__(self, args, num_classes=10):
        self.args = args
        super(Dense, self).__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, num_classes)
        Model_fn.__init__(self, args, loss_fn=torch.nn.CrossEntropyLoss())

    def forward(self, x):
        x = x['pixels']
        x = x.view(-1, 1 * 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def nl(x, y):
    return F.nll_loss(x.log(), y)
