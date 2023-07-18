import os
import sys
import yaml
import argparse
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

from models.ugan import UGAN_Nets, Gradient_Difference_Loss, Weights_Normal, Gradient_Penalty
from data.datasets import TrainingSet, TestValSet
from utils import seed_everything
from utils.log_config import LOGURU_FORMAT


## get configs and training options
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", type=str, default="configs/train_lsui3879.yaml")
parser.add_argument("--name", type=str, default="experiment", help="name for training process")
parser.add_argument("--epoch", type=int, default=0, help="which epoch to start from")
parser.add_argument("--iteration", type=int, default=0, help="which iteration to start from")
parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--l1_weight", type=float, default=100, help="Weight for L1 loss")
parser.add_argument("--ig_weight", type=float, default=0, help="0 for UGAN / 1 for UGAN-P")
parser.add_argument("--gp_weight", type=float, default=10, help="Weight for gradient penalty (D)")
parser.add_argument("--n_critic", type=int, default=5, help="training steps for D per iter w.r.t G")
parser.add_argument('--seed', type=int, default=2023, help='lucky random seed')
args = parser.parse_args()

## training params
train_name = args.name
start_epoch = args.epoch
num_epochs = args.num_epochs
batch_size =  args.batch_size
lr_rate = args.lr
num_critic = args.n_critic
seed = args.seed
lambda_gp = args.gp_weight # 10 (default)  
lambda_1 = args.l1_weight  # 100 (default) 
lambda_2 = args.ig_weight  # UGAN-P (default)
model_v = "UGAN_P" if lambda_2 else "UGAN"

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

# Initialize generator and discriminator
ugan_ = UGAN_Nets(base_model='pix2pix')
generator = ugan_.netG.to(DEVICE)
discriminator = ugan_.netD.to(DEVICE)

""" UGAN specifics: loss functions and patch-size
-------------------------------------------------"""
L1_G  = torch.nn.L1Loss().to(DEVICE) # l1 loss term
L1_gp = Gradient_Penalty().to(DEVICE) # wgan_gp loss term
L_gdl = Gradient_Difference_Loss().to(DEVICE) # GDL loss term

# Initialize weights or load pretrained models
if args.epoch == 0:
    generator.apply(Weights_Normal)
    discriminator.apply(Weights_Normal)
else:
    generator.load_state_dict(torch.load("checkpoints/%s/%s/generator_%d.pth" % (model_v, train_name, args.epoch-1)))
    discriminator.load_state_dict(torch.load("checkpoints/%s/%s/discriminator_%d.pth" % (model_v, train_name, args.epoch-1)))
    logger.info("Loaded model from epoch %d" %(args.epoch-1))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_rate)


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

# create a tensorboard writer
tensorboard_writer = SummaryWriter(log_dir=tensorboard_log_dir)

## Training pipeline
iteration_index = args.iteration
for epoch in range(start_epoch, start_epoch + num_epochs):
    for i, batch in enumerate(dataloader):
        # Model inputs
        imgs_distorted = batch["A"].to(DEVICE)
        imgs_good_gt = batch["B"].to(DEVICE)

        ## Train Discriminator
        optimizer_D.zero_grad()
        imgs_fake = generator(imgs_distorted)
        pred_real = discriminator(imgs_good_gt)
        pred_fake = discriminator(imgs_fake)
        loss_D_wgan = -torch.mean(pred_real) + torch.mean(pred_fake) # wgan 
        gradient_penalty = L1_gp(discriminator, imgs_good_gt.data, imgs_fake.data)
        loss_D = lambda_gp * gradient_penalty + loss_D_wgan # Eq.2 paper
        tensorboard_writer.add_scalar('Discriminator/loss_D_wgan', loss_D_wgan, iteration_index)
        tensorboard_writer.add_scalar('Discriminator/gradient_penalty', gradient_penalty, iteration_index)
        tensorboard_writer.add_scalar('Discriminator/loss_D', loss_D, iteration_index)
        loss_D.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        ## Train Generator at 1:num_critic rate 
        if i % num_critic == 0:
            imgs_fake = generator(imgs_distorted)
            pred_fake = discriminator(imgs_fake.detach())
            loss_gen = -torch.mean(pred_fake)
            loss_1 = L1_G(imgs_fake, imgs_good_gt)
            loss_gdl = L_gdl(imgs_fake, imgs_good_gt)
            # Total loss: Eq.6 in paper
            loss_G = loss_gen + lambda_1 * loss_1 + lambda_2 * loss_gdl
            tensorboard_writer.add_scalar('Generator/loss_gen', loss_gen, iteration_index)
            tensorboard_writer.add_scalar('Generator/loss_L1', loss_1, iteration_index)
            tensorboard_writer.add_scalar('Generator/loss_gdl', loss_gdl, iteration_index)
            tensorboard_writer.add_scalar('Generator/loss_G', loss_G, iteration_index)  
            loss_G.backward()
            optimizer_G.step()

        ## Print log
        if not (i % 50) or (i == len(dataloader)-1):
            logger.info("[iteration: %d] [Epoch %d/%d, batch %d/%d] [DLoss: %.3f, GLoss: %.3f]"
                              %(
                                iteration_index,
                                epoch, num_epochs, i, len(dataloader),
                                loss_D.item(), loss_G.item(),
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

