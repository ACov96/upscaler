import sys
import torch
import traceback
from torch import nn, optim, cuda
from torch.utils import data
from model import SRCNN, save_model, load_model
from util import print_flush, eprint

class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        return torch.sqrt(self.mse(x, y))
    
def train(training_data, dev_data, args):
    training_gen = data.DataLoader(training_data, batch_size=2)
    dev_gen = data.DataLoader(dev_data, batch_size=2)
    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    print('Initializing model')
    model = SRCNN()
    loss = RMSE()
    if cuda.device_count() > 1:
        print('Using %d CUDA devices' % cuda.device_count())
        model = nn.DataParallel(model, device_ids=[i for i in range(cuda.device_count())])
    model.to(device)
    loss.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    def _train(data, opt=True):
        total = 0
        for y, x in data:
            y, x = y.to(device), x.to(device)
            pred_y = model(x)
            l = loss(pred_y, y)
            total += l.item()
            if opt:
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
        cuda.synchronize()
        return total

    print('Training')
    for ep in range(args.ep):
        train_loss = _train(training_gen)
        dev_loss = _train(dev_gen, opt=False)
        print_flush('Epoch %d: Train %.4f Dev %.4f' % (ep, train_loss, dev_loss))
        if ep % 50 == 0:
            save_model(model, args.o)
    return model
