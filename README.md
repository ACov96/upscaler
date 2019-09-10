# Super Resolution CNN Upscaler

An SRCNN tool designed to upscale 1080p images to 4k. Built using PyTorch.

## Setup

Install dependencies

```bash
pip install -r requirements.txt
```

## Usage 

To train the model, the only required parameter is the `--data-path` parameter, which just points to a directory full of 4k images. The program will take care of generating a training and dev set at startup.

```bash
python main.py --data-path /path/to/data 
```

For more options, run this.
```bash
python main.py -h
```
