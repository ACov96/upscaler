import argparse
import torch
from glob import glob
from data import ImageDataset
from train import train
from model import save_model

def _parseArgs():
    parser = argparse.ArgumentParser(description='Deep Learning based upscaler tool.')
    parser.add_argument('--data-path', type=str, help='Path to data', required=True)
    parser.add_argument('-lr', type=float, help='Learning rate', default=0.01)
    parser.add_argument('-o', type=str, help='Path to write trained model out to', default='./out.model')
    parser.add_argument('-ep', type=int, help='Number of epochs to train', default=100)
    return parser.parse_args()

def main(args):
    files = glob(args.data_path + '*.jpg')
    training_data = ImageDataset(files[:int(len(files) * .9)])
    dev_data = ImageDataset(files[int(len(files) * .9):])
    model = train(training_data, dev_data, args)
    print('Saving model to %s' % args.o)
    save_model(model, args.o)

if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    main(_parseArgs())
