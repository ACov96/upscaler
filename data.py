import torch
from random import shuffle
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
from os.path import join

class ImageDataset(Dataset):
    def __init__(self, files):
        self.files = files
        self.transform = ToTensor()
        shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        im = Image.open(self.files[idx]).convert('RGB')
        im = im.resize((3840, 2160))
        rescaled = im.copy()
        basewidth = rescaled.size[0] // 2
        wpercent = (basewidth/float(rescaled.size[0]))
        hsize = int((float(rescaled.size[1])*float(wpercent)))
        rescaled = rescaled.resize((basewidth, hsize))
        rescaled = rescaled.resize((im.size[0], im.size[1]))
        im_t = self.transform(im)
        rescaled_t = self.transform(rescaled)
        return im_t, rescaled_t

