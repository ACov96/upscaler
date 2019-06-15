import torch
from torch import nn, save, load

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 10)
        self.conv2 = nn.Conv2d(8, 8, 1)
        self.conv3 = nn.ConvTranspose2d(8, 3, 10)
        self.act = nn.ReLU()

    def forward(self, x):
        first = self.act(self.conv1(x))
        second = self.act(self.conv2(first))
        third = self.act(self.conv3(second))
        return third

def save_model(m, path):
    try:
        state_dict = m.module.state_dict()
    except AttributeError:
        state_dict = m.state_dict()
    save(state_dict, path)
        
def load_model(path):
    m = SRCNN()
    m.load_state_dict(load(path))
    return m
