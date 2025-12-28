"""
Models module for Simple-FL

This module contains all neural network model architectures.
"""

# Import all models (keeping original imports)
# from .dnn import DNN
from .cnn import CNN_FEMNIST, ResNet18
from .aggregator import PA
from .din import din, DIN, DIN_mcc
from .dnn import DNN
# from .din_mcc import DIN_mcc
# from .gcn import GCN
# from .gpa import GPA
from .mpnn import PBA, PGA, GraphAvg
# from .pga_mp import PGAW
# import mpnn
from .model_base import *
from .model_fn import *
from .lenet5 import LeNet5, Dense
from .mlp import MLP_Mixer
from .resnet import *
