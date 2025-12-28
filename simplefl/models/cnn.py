from .model_base import *
from .model_fn import *
from torchvision.models import resnet18


class ResNet18(nn.Module, Model_fn):
    def __init__(self, args, num_classes=100):
        super(ResNet18, self).__init__()
        self.num_classes = num_classes
        self.args = args
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(2, 64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)
        Model_fn.__init__(self, args, loss_fn=torch.nn.CrossEntropyLoss())

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x['pixels']
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNN_FEMNIST(nn.Module, Model_fn):
    def __init__(self, args, num_classes=10):
        super(CNN_FEMNIST, self).__init__()
        self.num_classes = num_classes
        self.args = args
        
        # Default to 2layer if cnn_version is not in args
        cnn_version = getattr(args, 'cnn_version', '2layer')
        
        if cnn_version == "2layer":
            self.conv1 = nn.Conv2d(1, 32, 3)
            self.conv2 = nn.Conv2d(32, 64, 3)
            self.dense1 = nn.Linear(64 * 12 * 12, 128)
            self.dense2 = nn.Linear(128, num_classes)
        else:  # 3layer
            self.conv_layer = nn.Sequential(
                nn.Conv2d(1, 32, 3, 1, 1),  
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.fc_layer = nn.Sequential(
                nn.Linear(128 * 3 * 3, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )
        
        Model_fn.__init__(self, args, loss_fn=torch.nn.CrossEntropyLoss())

    def forward(self, x):
        x = x['pixels']
        if hasattr(self, 'conv_layer'):  # 3layer version
            x = self.conv_layer(x)
            x = self.fc_layer(x.flatten(1))
        else:  # 2layer version
            x = self.conv1(x)
            x = torch.relu(x)
            x = self.conv2(x)
            x = torch.relu(x)
            x = torch.max_pool2d(x, 2)
            x = x.flatten(1)
            x = self.dense1(x)
            x = torch.relu(x)
            x = self.dense2(x)
        return x


class LeNet5(nn.Module, Model_fn):
    def __init__(self, args, num_classes=10):

        super(LeNet5, self).__init__()
        self.num_classes = num_classes
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


class Dense(nn.Module, Model_fn):
    def __init__(self, args, num_classes=10):
        self.num_classes = num_classes
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


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(
                inchannel,
                outchannel,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(2, outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.GroupNorm(2, outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    inchannel, outchannel, kernel_size=1, stride=stride, bias=False
                ),
                nn.GroupNorm(2, outchannel),
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
