import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
import os
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
from glob import glob
from tqdm import tqdm

from torchvision.transforms.functional import normalize

import cv2
import argparse

from basicsr.archs.mambalfnet_arch import MambaLFNet
from basicsr.utils import imwrite, img2tensor, tensor2img, scandir


parser = argparse.ArgumentParser(description='Demo Low-light Image Enhancement')
parser.add_argument('--test_path', default='./datasets/LOLv1/test/low/', type=str, help='Input images')
parser.add_argument('--result_path', default='./results/LOLv1/', type=str, help='Directory for results')
parser.add_argument('--ckpt',
                    default='./checkpoints/LOLv1/models/model_bestPSNR.pth', type=str,
                    help='Path to weights')

args = parser.parse_args()


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

inp_dir = args.test_path
out_dir = args.result_path

os.makedirs(out_dir, exist_ok=True)

# Load corresponding models architecture and weights
model = MambaLFNet()
model.cuda()

load_checkpoint(model, args.ckpt)
model.eval()

print('restoring images......')

result_root = f'{args.result_path}/{os.path.basename(args.test_path)}'

mul = 16
index = 0
psnr_val_rgb = []
img_paths = sorted(list(scandir(args.test_path, suffix=('jpg', 'JPG', 'png', 'PNG', 'bmp'), recursive=True, full_path=True)))
for img_path in tqdm(img_paths):
    img_name = img_path.replace(args.test_path+'/', '')
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_t = img2tensor(img / 255, bgr2rgb=False, float32=True)
    img_t = img_t.unsqueeze(0).cuda()
    

    # Pad the input if not_multiple_of 16
    h, w = img_t.shape[2], img_t.shape[3]
    H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
    padh = H - h if h % mul != 0 else 0
    padw = W - w if w % mul != 0 else 0
    img_t = F.pad(img_t, (0, padw, 0, padh), 'reflect')
    with torch.no_grad():
        restored = model(img_t)

    restored = torch.clamp(restored, 0, 1)
    restored = restored[:, :, :h, :w]
    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored = img_as_ubyte(restored[0])

    output = restored.astype('uint8')
    
    # save restored img
    save_restore_path = img_path.replace(args.test_path, result_root)
    imwrite(output, save_restore_path)

print(f"Files saved at {out_dir}")
print('finish !')
