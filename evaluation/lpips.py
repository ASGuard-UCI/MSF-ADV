import lpips
import torch

# Implementation from The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
# https://github.com/richzhang/PerceptualSimilarity

loss_fn_alex = lpips.LPIPS(net='alex')
# loss_fn_vgg = lpips.LPIPS(net='vgg')

img0 = torch.zeros(1,3,1920,1080) # image should be RGB, IMPORTANT: normalized to [-1,1]
img1 = torch.zeros(1,3,1920,1080)

print(loss_fn_alex(img0,img1))