from .model_base import *
from .model_fn import *


class Stem(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Stem, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.Linear1 = nn.Linear(num_inputs, num_outputs)

        # Width and height of each patch
        self.kw = 7
        self.kh = 7
        # Stride until next patch
        self.stride = 7

    def forward(self, x):

        # Creating 16 patches of 7*7 size each using unfold method
        patches = x.unfold(2, self.kw, self.stride).unfold(
            3, self.kh, self.stride)

        # Flattening each patch to create a vector
        patches = torch.flatten(patches, start_dim=4)

        # Reshaping to form a tensor of size 16*49
        patches = patches.reshape(patches.shape[0], 1, -1, 49)

        # Passing the patches to a linear layer
        out = self.Linear1(patches)
        return out


class Backbone(torch.nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, num_input_1,
                 num_hidden_1, num_outputs_1):
        super(Backbone, self).__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.num_hidden_1 = num_hidden_1
        self.num_outputs_2 = num_outputs_2
        self.Linear1 = nn.Linear(num_inputs,  num_hidden)
        self.relu = nn.ReLU()
        self.Linear2 = nn.Linear(num_hidden,  num_outputs)
        self.Linear3 = nn.Linear(num_input_1,  num_hidden_1)
        self.Linear4 = nn.Linear(num_hidden_1,  num_outputs_1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):

        # First MLP

        # Transposing X on the 2nd and 3rd dimension of width and height of the image
        x = torch.transpose(x, 2, 3)
        linear1 = self.Linear1(x)
        activation = self.relu(linear1)
        out1 = self.Linear2(activation)
        O1 = self.dropout(out1)

        # Tranposing the output from first MLP
        out1 = torch.transpose(O1, 2, 3)

        # Second MLP
        linear2 = self.Linear3(out1)
        activation = self.relu(linear2)
        out2 = self.Linear4(activation)
        O2 = self.dropout(out2)

        return O2


class Classifier(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Classifier, self).__init__()
        self.Linear1 = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):

        # Mean feature
        x = x.mean(axis=2)
        x = x.reshape(x.shape[0], -1)

        # Linear Layer
        out = self.Linear1(x)
        return out


s_num_inputs, s_num_outputs = 49, 100
num_inputs_1, num_hidden_1, num_outputs_1, num_input_2, num_hidden_2, num_outputs_2 = 16, 200, 100, 100, 150, 16
c_num_inputs, c_num_outputs = 16, 10


class MLP_Mixer(nn.Module, Model_fn):
    def __init__(self,args,num_classes=10):
        super(MLP_Mixer, self).__init__()
        self.net = nn.Sequential(Stem(s_num_inputs, s_num_outputs),
                                Backbone(num_inputs_1, num_hidden_1, num_outputs_1,
                                                num_input_2, num_hidden_2, num_outputs_2),
                                 Classifier(c_num_inputs, num_classes))
        Model_fn.__init__(self, args, loss_fn=torch.nn.CrossEntropyLoss())
    def forward(self, x):
        x=x['pixels']
        out=self.net(x)
        return out
