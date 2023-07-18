# py libs
import os
import sys
import yaml
import argparse
import torch
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
from loguru import logger
from kornia.losses import SSIMLoss
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.raune_net import RauneNet
from models.utils import Weights_Normal
from models.loss_fn import SemanticContentLoss
from data.datasets import TrainingSet, TestValSet
from utils import seed_everything
from utils.log_config import LOGURU_FORMAT


# Command-line options and arguments
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", type=str, default="configs/train_lsui3879.yaml")
parser.add_argument("--name", type=str, default="experiment", help="name for training process")
parser.add_argument("--epoch", type=int, default=0, help="which epoch to start from")
parser.add_argument("--iteration", type=int, default=0, help="which iteration to start from")
parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate for adam optimizer")
parser.add_argument("--pcont_weight", type=float, default=1, help="Weight for L1/MAE loss")
parser.add_argument("--ssim_weight", type=float, default=1, help="Weight for ssim loss")
parser.add_argument("--scont_weight", type=float, default=0, help="Weight for semantic content loss")
parser.add_argument('--seed', type=int, default=2023, help='lucky random seed')
parser.add_argument('--num_down', type=int, default=2, help='number of downsampling')
parser.add_argument('--num_blocks', type=int, default=3, help='number of residual blocks')
parser.add_argument('--use_att_up', action='store_true', help='whether to use attention module in upsampling')
parser.add_argument('--use_lr_scheduler', action='store_true', help='whether to use learning rate scheduler')
args = parser.parse_args()

# Training params
train_name = args.name
start_epoch = args.epoch
num_epochs = args.num_epochs
batch_size =  args.batch_size
learning_rate = args.lr
seed = args.seed
num_down = args.num_down
num_blocks = args.num_blocks
use_att_up = args.use_att_up
use_lr_scheduler = args.use_lr_scheduler
lambda_pcont = args.pcont_weight     
lambda_ssim = args.ssim_weight  
lambda_scont = args.scont_weight
model_v = "RAUNENet"

# Get infomation from the data config file
with open(args.cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
dataset_path = cfg["dataset_path"]
folder_A = cfg["folder_A"]
folder_B = cfg["folder_B"]
validation_dir = cfg["validation_dir"]
channels = cfg["chans"]
img_width = cfg["im_width"]
img_height = cfg["im_height"] 
val_interval = cfg["val_interval"]
ckpt_interval = cfg["ckpt_interval"]

# Create some useful directories
samples_dir = "samples/{}/{}".format(model_v, train_name)
checkpoint_dir = "checkpoints/{}/{}/".format(model_v, train_name)
log_dir = os.path.join(checkpoint_dir, 'logs')
tensorboard_log_dir = os.path.join('runs', f"{model_v}/{train_name}")
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize logger
logger.remove(0)
logger.add(sys.stdout, format=LOGURU_FORMAT)
logger.add(os.path.join(log_dir, "train_{time}.log"), format=LOGURU_FORMAT)

# Write some training infomation into log file
logger.info(f"Starting Training Process...")
logger.info(f"model_version: {model_v}")
logger.info(f"samples_dir: {samples_dir}")
logger.info(f"checkpoint_dir: {checkpoint_dir}")
logger.info(f"log_dir: {log_dir}")
logger.info(f"tensorboard_log_dir: {tensorboard_log_dir}")
for option, value in vars(args).items():
    logger.info(f"{option}: {value}")
for option, value in cfg.items():
    logger.info(f"{option}: {value}")

# Set random seed
seed_everything(seed)

# Set device for pytorch
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# Loss function
L_pixel_content  = torch.nn.L1Loss().to(DEVICE)       # pixel content loss
L_ssim = SSIMLoss(11).to(DEVICE)                      # structural similarity loss
L_semantic_content = SemanticContentLoss().to(DEVICE) # semantic content loss (features extract from VGG_19_BN)

# Initialize model
model = RauneNet(channels, 3, num_blocks, num_down, use_att_up=use_att_up).to(DEVICE)

# Initialize weights or load pretrained models
if args.epoch == 0:
    model.apply(Weights_Normal)
else:
    model.load_state_dict(torch.load("checkpoints/%s/%s/weights_%d.pth" % (model_v, train_name, args.epoch-1)))
    logger.info("Loaded model from epoch %d" %(args.epoch-1))

# Optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate schedulers
if use_lr_scheduler:
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75, 100], gamma=0.1)

# Data pipeline
transforms_ = [
    transforms.Resize((img_height, img_width), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

train_dataloader = DataLoader(
    TrainingSet(dataset_path, folder_A, folder_B, transforms_=transforms_),
    batch_size = batch_size,
    shuffle = True,
    num_workers = 8,
)

val_dataloader = DataLoader(
    TestValSet(dataset_path, transforms_=transforms_, sub_dir=validation_dir),
    batch_size=4,
    shuffle=True,
    num_workers=1,
)

# Create a tensorboard writer
tensorboard_writer = SummaryWriter(log_dir=tensorboard_log_dir)

# Training pipeline
iteration_index = args.iteration
for epoch in range(start_epoch, start_epoch + num_epochs):
    for i, batch in enumerate(train_dataloader):
        # model inputs
        imgs_input = batch["A"].to(DEVICE)
        imgs_gt = batch["B"].to(DEVICE)

        optimizer.zero_grad()
        imgs_enhanced = model(imgs_input)
        loss_pcont = L_pixel_content(imgs_enhanced, imgs_gt)
        loss_ssim = L_ssim(imgs_enhanced, imgs_gt)
        loss_scont = L_semantic_content(imgs_enhanced, imgs_gt)
        # total loss
        loss_total = lambda_pcont * loss_pcont + lambda_ssim * loss_ssim + lambda_scont * loss_scont
        # write into tensorboard
        tensorboard_writer.add_scalar('model/loss_pcont', loss_pcont, iteration_index)
        tensorboard_writer.add_scalar('model/loss_ssim', loss_ssim, iteration_index)
        tensorboard_writer.add_scalar('model/loss_scont', loss_scont, iteration_index)
        tensorboard_writer.add_scalar('model/loss_total', loss_total, iteration_index)  
        loss_total.backward()
        optimizer.step()

        # print log
        if (i%50 == 0) or (i == len(train_dataloader)-1):
            logger.info("[iteration: {:d}, lr: {:f}] [Epoch {:d}/{:d}, batch {:d}/{:d}] [Loss: {:.3f}]".format(
                iteration_index, optimizer.param_groups[0]['lr'],
                epoch, args.epoch + num_epochs-1, i, len(train_dataloader)-1,
                loss_total.item()
            ))

        # if at sample interval save image
        batches_done = epoch * len(train_dataloader) + i
        if batches_done % val_interval == 0:
            val_batch = next(iter(val_dataloader))
            imgs_val = val_batch["val"].to(DEVICE)
            imgs_gen = model(imgs_val)
            img_sample = torch.cat((imgs_val.data, imgs_gen.data), -2)
            save_image(img_sample, "samples/{}/{}/{}.png".format(model_v, train_name, batches_done), nrow=4, normalize=True)
        iteration_index += 1
    # learning rate schedule
    if use_lr_scheduler:
        scheduler.step()
    # save model checkpoints
    if (epoch % ckpt_interval == 0) or (epoch == args.epoch+num_epochs-1):
        torch.save(model.state_dict(), "checkpoints/{}/{}/weights_{:d}.pth".format(model_v, train_name, epoch))