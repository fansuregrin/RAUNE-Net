import os
import time
import argparse
import sys
from loguru import logger
from os.path import join, exists, basename
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from models.waternet import WaterNet
from data.datasets import WaterNetTestValSet
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
opt = parser.parse_args()

model_v = 'WaterNet'

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

# Set device for pytorch
test_ds = WaterNetTestValSet(
    raw_dir = opt.data_dir,
    img_width =  opt.input_width,
    img_height = opt.input_height
)
test_dl = DataLoader(test_ds, batch_size=1)

# Initialize model
model = WaterNet().to(DEVICE)
model.load_state_dict(torch.load(model_path))
model.eval()
logger.info(f"Loaded model from {model_path}")

# Testing loop
total_time = 0.0
for batch in test_dl:
    # prepare input images
    raw_imgs = batch['raw'].to(DEVICE)
    wb_imgs = batch['wb'].to(DEVICE)
    gc_imgs = batch['gc'].to(DEVICE)
    he_imgs = batch['he'].to(DEVICE)
    # generate enhanced image
    start_t = time.time()
    with torch.no_grad():
        gen_imgs = model(raw_imgs, wb_imgs, he_imgs, gc_imgs)
    total_time += (time.time()-start_t)
    # save output image
    imgs_paired = torch.cat((raw_imgs.data, gen_imgs.data), -1)
    for (raw_img, gen_img, img_paired, img_path) in zip(raw_imgs, gen_imgs, imgs_paired, batch['raw_path']):
        save_image(img_paired, join(opt.result_dir, opt.test_name, 'paired', basename(img_path)))
        save_image(raw_img.data, os.path.join(opt.result_dir, opt.test_name, 'single/input', basename(img_path)))
        save_image(gen_img.data, os.path.join(opt.result_dir, opt.test_name, 'single/predicted', basename(img_path)))
    logger.info(f"Tested: {img_path}")

# Output summary logs
logger.info("Total samples: {:d}".format(len(test_ds)))
avg_time = total_time / len(test_ds)
logger.info("Time taken: {:.3f} sec at {:.3f} fps".format(total_time, 1./avg_time))
logger.info("Saved enhanced images in {}".format(opt.result_dir))