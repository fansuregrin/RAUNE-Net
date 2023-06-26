import os
import time
import argparse
import numpy as np
import sys
import torch
from loguru import logger
from PIL import Image
from os.path import join, exists, basename
from torchvision.utils import save_image
import torchvision.transforms as transforms

from models.raune_net import RauneNet
from data import is_image_file
from utils.log_config import LOGURU_FORMAT


# Command-line options and arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, help='name for checkpoint')
parser.add_argument('--test_name', type=str, help='name for test dataset')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='path to checkpoint dir')
parser.add_argument("--data_dir", type=str, help='test dataset dir')
parser.add_argument("--result_dir", type=str, default="results")
parser.add_argument('--epoch', type=int, default=95, help='which epoch to load')
parser.add_argument('--input_width', type=int, default=256)
parser.add_argument('--input_height', type=int, default=256)
parser.add_argument('--num_down', type=int, default=2, help='number of downsampling in resnet model')
parser.add_argument('--num_blocks', type=int, default=30, help='number of residual blocks in resnet model')
parser.add_argument('--use_att_up', action='store_true', help='whether to use attention module in upsampling')
opt = parser.parse_args()

model_v = 'RAUNENet'

# Check the path of trained model
model_path = os.path.join(opt.checkpoint_dir, model_v, opt.name, f'weights_{opt.epoch}.pth')
assert exists(model_path), "model weights not found"

# Make useful directories for saving results
os.makedirs(os.path.join(opt.result_dir, opt.test_name, 'paired'), exist_ok=True)
os.makedirs(os.path.join(opt.result_dir, opt.test_name, 'single/input'), exist_ok=True)
os.makedirs(os.path.join(opt.result_dir, opt.test_name, 'single/predicted'), exist_ok=True)

# Initialize logger
log_dir = os.path.join(opt.checkpoint_dir, model_v, opt.name, 'logs')
logger.remove(0)
logger.add(sys.stdout, format=LOGURU_FORMAT)
logger.add(os.path.join(log_dir, "test_{time}.log"), format=LOGURU_FORMAT)

# Write useful logs
logger.info(f"Starting Test Process...")
for option, value in vars(opt).items():
    logger.info(f"{option}: {value}")
logger.info(f"model_path: {model_path}")

# Set device for pytorch
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# Set data pipeline
img_width, img_height, channels = opt.input_width, opt.input_height, 3
transforms_ = [transforms.Resize((img_height, img_width), transforms.InterpolationMode.BICUBIC),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
transform = transforms.Compose(transforms_)
test_files = []
for rel_path in os.listdir(opt.data_dir):
    path = os.path.join(opt.data_dir, rel_path)
    if is_image_file(path):
        test_files.append(path)
test_files.sort()

# Initialize model
model = RauneNet(channels, 3, opt.num_blocks, opt.num_down, use_att_up=opt.use_att_up).to(DEVICE)
model.load_state_dict(torch.load(model_path))
model.eval()
logger.info(f"Loaded model from {model_path}")

# Testing loop
times = []
for path in test_files:
    # prepare input image
    inp_img = transform(Image.open(path))
    inp_img = inp_img.unsqueeze(0).to(DEVICE)
    # generate enhanced image
    s = time.time()
    with torch.no_grad():
        gen_img = model(inp_img)
    times.append(time.time()-s)
    # save output image
    img_sample_paired = torch.cat((inp_img.data, gen_img.data), -1)
    save_image(img_sample_paired, join(opt.result_dir,    opt.test_name, 'paired', basename(path)), normalize=True)
    save_image(inp_img.data, os.path.join(opt.result_dir, opt.test_name, 'single/input', basename(path)), normalize=True)
    save_image(gen_img.data, os.path.join(opt.result_dir, opt.test_name, 'single/predicted', basename(path)), normalize=True)
    logger.info(f"Tested: {path}")

# Output summary logs
if (len(times) > 1):
    logger.info("Total samples: {:d}".format(len(test_files))) 
    total_time, mean_time = np.sum(times), np.mean(times)
    logger.info("Time taken: {:.3f} sec at {:.3f} fps".format(total_time, 1./mean_time))
    logger.info("Saved enhanced images in {}\n".format(opt.result_dir))