import os
import sys
import yaml
import argparse
import torch
from loguru import logger
from kornia.losses import SSIMLoss
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.waternet import WaterNet
from models.loss_fn import SemanticContentLoss
from data.datasets import WaterNetTrainSet, WaterNetTestValSet
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
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--mae_weight", type=float, default=1, help="Weight for L1/MAE loss")
parser.add_argument("--ssim_weight", type=float, default=0, help="Weight for ssim loss")
parser.add_argument("--scont_weight", type=float, default=0, help="Weight for semantic content loss")
parser.add_argument('--seed', type=int, default=2023, help='lucky random seed')
args = parser.parse_args()

# Training params
train_name = args.name
start_epoch = args.epoch
num_epochs = args.num_epochs
batch_size =  args.batch_size
learning_rate = args.lr
seed = args.seed
lambda_mae = args.mae_weight
lambda_ssim = args.ssim_weight
lambda_scont = args.scont_weight
model_v = "WaterNet"

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
samples_dir = "samples/%s/%s" % (model_v, train_name)
checkpoint_dir = "checkpoints/%s/%s/" % (model_v, train_name)
log_dir = os.path.join(checkpoint_dir, 'logs')
tensorboard_log_dir = os.path.join('waternet_runs', f"{model_v}_{train_name}")
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(tensorboard_log_dir, exist_ok=True)

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
L_mae  = torch.nn.L1Loss().to(DEVICE)                 # l1 loss or mean absoluate error
L_ssim = SSIMLoss(11).to(DEVICE)                      # ssim loss
L_semantic_content = SemanticContentLoss().to(DEVICE) # semantic content loss (features extract from VGG-19-BN)

# Initialize model
model = WaterNet().to(DEVICE)

# Initialize weights or load pretrained models
if args.epoch == 0:
    pass
else:
    model.load_state_dict(torch.load("checkpoints/%s/%s/weights_%d.pth" % (model_v, train_name, args.epoch-1)))
    logger.info("Loaded model from epoch %d" %(args.epoch-1))

# Optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Data pipeline
train_set = WaterNetTrainSet(
    raw_dir=os.path.join(dataset_path, folder_A),
    ref_dir=os.path.join(dataset_path, folder_B),
    img_height=img_height,
    img_width=img_width
)

val_set = WaterNetTestValSet(
    raw_dir=os.path.join(dataset_path, validation_dir),
    img_height=img_height,
    img_width=img_width
)

train_dataloader = DataLoader(
    train_set,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 8,
)

val_dataloader = DataLoader(
    val_set,
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
        raw_imgs = batch['raw'].to(DEVICE)
        ref_imgs = batch['ref'].to(DEVICE)
        wb_imgs = batch['wb'].to(DEVICE)
        gc_imgs = batch['gc'].to(DEVICE)
        he_imgs = batch['he'].to(DEVICE)

        optimizer.zero_grad()
        predicted_imgs = model(raw_imgs, wb_imgs, he_imgs, gc_imgs)
        loss_mae = L_mae(predicted_imgs, ref_imgs)
        loss_ssim = L_ssim(predicted_imgs, ref_imgs)
        loss_scont = L_semantic_content(predicted_imgs, ref_imgs)
        # total loss
        loss_total = lambda_mae * loss_mae + lambda_ssim * loss_ssim + lambda_scont * loss_scont
        # write into tensorboard
        tensorboard_writer.add_scalar('model/loss_mae', loss_mae, iteration_index)
        tensorboard_writer.add_scalar('model/loss_ssim', loss_ssim, iteration_index)
        tensorboard_writer.add_scalar('model/loss_scont', loss_scont, iteration_index)
        tensorboard_writer.add_scalar('model/loss_total', loss_total, iteration_index)  
        loss_total.backward()
        optimizer.step()

        # print log
        if (i%50 == 0) or (i == len(train_dataloader)-1):
            logger.info("[iteration: %d] [Epoch %d/%d: batch %d/%d] [Loss: %.3f]"
                        %(
                        iteration_index, epoch, args.epoch+num_epochs-1, i, len(train_dataloader)-1, loss_total.item(),
                        )
            )
        # if at sample interval save image
        batches_done = epoch * len(train_dataloader) + i
        if batches_done % val_interval == 0:
            val_batch = next(iter(val_dataloader))
            val_raw_imgs = val_batch["raw"].to(DEVICE)
            val_wb_imgs = val_batch["wb"].to(DEVICE)
            val_he_imgs = val_batch["he"].to(DEVICE)
            val_gc_imgs = val_batch["gc"].to(DEVICE)
            val_pred_imgs = model(val_raw_imgs, val_wb_imgs, val_he_imgs, val_gc_imgs)
            img_sample = torch.cat((val_raw_imgs.data, val_pred_imgs.data), -2)
            save_image(img_sample, "samples/%s/%s/%s.png" % (model_v, train_name, batches_done), nrow=4)
        iteration_index += 1

    # save model checkpoints
    if (epoch % ckpt_interval == 0) or (epoch == args.epoch+num_epochs-1):
        torch.save(model.state_dict(), "checkpoints/%s/%s/weights_%d.pth" % (model_v, train_name, epoch))