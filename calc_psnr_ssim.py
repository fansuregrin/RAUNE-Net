import argparse
import os
import numpy as np
from torchvision.transforms.functional import to_tensor
from PIL import Image
from kornia.metrics import psnr, ssim
from data import is_image_file


parser = argparse.ArgumentParser(prog='Calculating PSNR and SSIM')
parser.add_argument('--input_dir', type=str, help='path to folder of input images')
parser.add_argument('--refer_dir', type=str, help='path to folder of reference images')
parser.add_argument('--output_dir', type=str, help='path to folder of results')
parser.add_argument('--use_log', action='store_false', help='whether print log to std ouput')
parser.add_argument('--resize', action='store_true', help='whether resize the input and reference images')
parser.add_argument('--width', default=256, type=int, help='image width for resizing')
parser.add_argument('--height', default=256, type=int, help='image height for resizing')
args = parser.parse_args()

input_dir, refer_dir = args.input_dir, args.refer_dir
input_image_paths = []
refer_image_paths = []
for filename in os.listdir(input_dir):
    full_path = os.path.join(input_dir, filename)
    if is_image_file(full_path):
        input_image_paths.append(full_path)
for filename in os.listdir(refer_dir):
    full_path = os.path.join(refer_dir, filename)
    if is_image_file(full_path):
        refer_image_paths.append(full_path)
input_image_paths.sort()
refer_image_paths.sort()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

assert len(input_image_paths) == len(refer_image_paths)
psnr_values = []
ssim_values = []
f = open(os.path.join(output_dir, 'quantitive_eval.csv'), 'w')
f.write('img_name,psnr,ssim\n')
for input_img_path, refer_img_path in zip(input_image_paths, refer_image_paths):
    input_img = Image.open(input_img_path)
    refer_img = Image.open(refer_img_path)
    if args.resize:
        input_img = input_img.resize((args.width, args.height))
        refer_img = refer_img.resize((args.width, args.height))
    assert input_img.size == refer_img.size
    input_img = to_tensor(input_img).unsqueeze(0)
    refer_img = to_tensor(refer_img).unsqueeze(0)
    psnr_value = psnr(input_img, refer_img, max_val=1.0).item()
    ssim_value = ssim(input_img, refer_img, max_val=1.0, window_size=11).mean().item()
    psnr_values.append(psnr_value)
    ssim_values.append(ssim_value)
    f.write(f"{os.path.basename(input_img_path)},{psnr_value:.3f},{ssim_value:.3f}\n")
    if args.use_log:
        print(f"{os.path.basename(input_img_path)}, psnr: {psnr_value:.3f}, ssim: {ssim_value:.3f}")
f.write(f"average_value,{np.average(psnr_values):.3f},{np.average(ssim_values):.3f}")
if args.use_log:
    print(f"average_value, psnr: {np.average(psnr_values):.3f}, ssim: {np.average(ssim_values):.3f}")