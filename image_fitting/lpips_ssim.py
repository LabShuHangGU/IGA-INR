from skimage import io
from skimage.metrics import structural_similarity as ssim
import lpips
import torch
from PIL import Image
import os
import torchvision.transforms as transforms
import re
import numpy as np
import pandas as pd
# from torchmetrics.image.ssim import MultiscaleStructuralSimilarityIndexMeasure  
from pytorch_msssim import ms_ssim  


# /data0/home/shikexuan/kodim_photo/
image_set=['01','02','03','04','05','06','07','08']
result_folder = 'results/'
results = []
name='sin_fr'



for i in image_set:
    print('Kodak'+i)
    gt = io.imread('/data0/home/shikexuan/kodim_photo/kodim'+i+'.png')
    # pattern = f'kodim{i}_sin_3layers_IGA_xuhao_[0-9]+_[0-9]+_[0-9]+_psnr_\d+\.\d+\.png'
    pattern = f'kodim{i}_sin\+fr_3layers_psnr_\d+\.\d+\.png'
    # pattern = f'kodim{i}_relu\+pe\+bn_3layers_psnr_\d+\.\d+\.png'
    for file_name in os.listdir(result_folder):
        if re.match(pattern, file_name):
            image_path = os.path.join(result_folder, file_name)
            image = io.imread(image_path)
            print(f'Reading: {image_path}')
            break  # Break the loop after finding the corresponding file.
    # Convert both images to grayscale mode for SSIM calculation.
    gt = np.transpose(gt, (2, 0, 1))  # Transform to (channels, height, width)
    image = np.transpose(image, (2, 0, 1))  # Transform to (channels, height, width) format.
    ssim_value = ssim(gt, image, multichannel=True,channel_axis=0)
    print(f'SSIM: {ssim_value}')

    transform = transforms.ToTensor()
    gt_tensor = transform(Image.open('/data0/home/shikexuan/kodim_photo/kodim' + i + '.png')).unsqueeze(0)
    image_tensor = transform(Image.open(image_path)).unsqueeze(0)
    ms_ssim_value = ms_ssim(gt_tensor, image_tensor, data_range=1.0).item()  # data_range=1.0
    print(f'MS-SSIM: {ms_ssim_value}')

    loss_fn = lpips.LPIPS(net='alex')
    transform = transforms.ToTensor()
    img1 = transform(Image.open('/data0/home/shikexuan/kodim_photo/kodim'+i+'.png')).unsqueeze(0)
    img2 = transform(Image.open(image_path)).unsqueeze(0)
    lpips_value = loss_fn(img1, img2)
    print(f'LPIPS: {lpips_value.item()}')
    print('------------')
    results.append({
        'Image': f'kodim{i}.png',
        'MS-SSIM': ms_ssim_value,
        'SSIM': ssim_value,
        'LPIPS': lpips_value.item()
    })


results_df = pd.DataFrame(results)
results_df.to_csv('metrics_'+name+'.csv', index=False)