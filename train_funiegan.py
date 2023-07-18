import os
import sys
import yaml
import argparse
import torch
import torchvision.transforms as transforms
from loguru import logger
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.funiegan import Weights_Normal, VGG19_PercepLoss, GeneratorFunieGAN, DiscriminatorFunieGAN
from data.datasets import TrainingSet, TestValSet
from utils import seed_everything
from utils.log_config import LOGURU_FORMAT


## get configs and training options
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", type=str, default="configs/train_lsui3879.yaml")
parser.add_argument("--name", type=str, default="experiment", help="name for training process")
parser.add_argument("--epoch", type=int, default=0, help="which epoch to start from")
parser.add_argument("--iteration", type=int, default=0, help="which iteration to start from")
parser.add_argument("--num_epochs", type=int, default=201, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of 1st order momentum")
parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of 2nd order momentum")
parser.add_argument('--seed', type=int, default=2023, help='lucky random seed')
args = parser.parse_args()

## training params
train_name = args.name
start_epoch = args.epoch
num_epochs = args.num_epochs
batch_size =  args.batch_size
lr_rate, lr_b1, lr_b2 = args.lr, args.b1, args.b2
seed = args.seed
model_v = 'FUnIE_GAN'

# load the data config file
with open(args.cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
# get info from config file
dataset_path = cfg["dataset_path"]
folder_A = cfg["folder_A"]
folder_B = cfg["folder_B"]
validation_dir = cfg["validation_dir"]
channels = cfg["chans"]
img_width = cfg["im_width"]
img_height = cfg["im_height"] 
val_interval = cfg["val_interval"]
ckpt_interval = cfg["ckpt_interval"]


## create some useful directories
samples_dir = "samples/%s/%s" % (model_v, train_name)
checkpoint_dir = "checkpoints/%s/%s/" % (model_v, train_name)
log_dir = os.path.join(checkpoint_dir, 'logs')
tensorboard_log_dir = os.path.join('runs', f"{model_v}/{train_name}")
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# initialize logger
logger.remove(0)
logger.add(sys.stdout, format=LOGURU_FORMAT)
logger.add(os.path.join(log_dir, "train_{time}.log"), format=LOGURU_FORMAT)

# write some training infomation into log file
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

# set random seed
seed_everything(seed)

# set device
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

""" FunieGAN specifics: loss functions and patch-size
-----------------------------------------------------"""
Adv_cGAN = torch.nn.MSELoss().to(DEVICE)
L1_G  = torch.nn.L1Loss().to(DEVICE)        # similarity loss (l1)
L_vgg = VGG19_PercepLoss().to(DEVICE)       # content loss (vgg)
lambda_1, lambda_con = 7, 3                 # 7:3 (as in paper)
patch = (1, img_height//16, img_width//16)  # 16x16 for 256x256

# Initialize generator and discriminator
generator = GeneratorFunieGAN().to(DEVICE)
discriminator = DiscriminatorFunieGAN().to(DEVICE)

# Initialize weights or load pretrained models
if args.epoch == 0:
    generator.apply(Weights_Normal)
    discriminator.apply(Weights_Normal)
else:
    generator.load_state_dict(torch.load("checkpoints/FunieGAN/%s/generator_%d.pth" % (train_name, args.epoch-1)))
    discriminator.load_state_dict(torch.load("checkpoints/FunieGAN/%s/discriminator_%d.pth" % (train_name, args.epoch-1)))
    logger.info("Loaded model from epoch %d" %(args.epoch-1))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))


## Data pipeline
transforms_ = [
    transforms.Resize((img_height, img_width), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
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

tensorboard_writer = SummaryWriter(log_dir=tensorboard_log_dir)

## Training pipeline
iteration_index = args.iteration
for epoch in range(start_epoch, start_epoch + num_epochs):
    for i, batch in enumerate(dataloader):
        # Model inputs
        imgs_distorted = batch["A"].to(DEVICE)
        imgs_good_gt = batch["B"].to(DEVICE)
        # Adversarial ground truths
        valid = torch.ones((imgs_distorted.size(0), *patch), requires_grad=False).to(DEVICE)
        fake = torch.zeros((imgs_distorted.size(0), *patch), requires_grad=False).to(DEVICE)

        ## Train Discriminator
        optimizer_D.zero_grad()
        imgs_fake = generator(imgs_distorted)
        pred_real = discriminator(imgs_good_gt, imgs_distorted)
        loss_real = Adv_cGAN(pred_real, valid)
        pred_fake = discriminator(imgs_fake, imgs_distorted)
        loss_fake = Adv_cGAN(pred_fake, fake)
        # Total loss: real + fake (standard PatchGAN)
        loss_D = 0.5 * (loss_real + loss_fake) * 10.0 # 10x scaled for stability
        # write into tensorboard
        tensorboard_writer.add_scalar('Discriminator/loss_real', loss_real, iteration_index)
        tensorboard_writer.add_scalar('Discriminator/loss_fake', loss_fake, iteration_index)
        tensorboard_writer.add_scalar('Discriminator/loss_D', loss_D, iteration_index)
        loss_D.backward()
        optimizer_D.step()

        ## Train Generator
        optimizer_G.zero_grad()
        imgs_fake = generator(imgs_distorted)
        pred_fake = discriminator(imgs_fake, imgs_distorted)
        loss_GAN =  Adv_cGAN(pred_fake, valid) # GAN loss
        loss_1 = L1_G(imgs_fake, imgs_good_gt) # similarity loss
        loss_con = L_vgg(imgs_fake, imgs_good_gt)# content loss
        # Total loss (Section 3.2.1 in the paper)
        loss_G = loss_GAN + lambda_1 * loss_1  + lambda_con * loss_con
        # write into tensorboard
        tensorboard_writer.add_scalar('Generator/loss_GAN', loss_GAN, iteration_index)
        tensorboard_writer.add_scalar('Generator/loss_L1', loss_1, iteration_index)
        tensorboard_writer.add_scalar('Generator/loss_Content', loss_con, iteration_index)
        tensorboard_writer.add_scalar('Generator/loss_G', loss_G, iteration_index)
        loss_G.backward()
        optimizer_G.step()

        ## Print log
        if not (i % 50) or (i == len(dataloader)-1):
            logger.info("[iteration: %d] [Epoch %d/%d: batch %d/%d] [DLoss: %.3f, GLoss: %.3f, AdvLoss: %.3f]"
                              %(
                                iteration_index,
                                epoch, num_epochs, i, len(dataloader),
                                loss_D.item(), loss_G.item(), loss_GAN.item(),
                               )
            )
        ## If at sample interval save image
        if iteration_index % val_interval == 0:
            imgs = next(iter(val_dataloader))
            imgs_val = imgs["val"].to(DEVICE)
            imgs_gen = generator(imgs_val)
            img_sample = torch.cat((imgs_val.data, imgs_gen.data), -2)
            save_image(img_sample, "samples/%s/%s/%s.png" % (model_v, train_name, iteration_index), nrow=4, normalize=True)
        iteration_index += 1

    ## Save model checkpoints
    if (epoch % ckpt_interval == 0) or (epoch == args.epoch+num_epochs-1):
        torch.save(generator.state_dict(), "checkpoints/%s/%s/generator_%d.pth" % (model_v, train_name, epoch))
        torch.save(discriminator.state_dict(), "checkpoints/%s/%s/discriminator_%d.pth" % (model_v, train_name, epoch))
