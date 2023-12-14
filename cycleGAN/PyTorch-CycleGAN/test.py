#!/usr/bin/python3

import argparse
import sys
import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import Generator
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/shoeprint/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B_police_400_400.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A_300.pth', help='B2A generator checkpoint file')
opt = parser.parse_args()
# print(opt)


# if torch.cuda.is_available() and not opt.cuda:
#     print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

print('네트워크 로드 완료')

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

print('input메모리 할당 A:', input_A.shape, 'B:', input_B.shape)

# Dataset loader
transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'), 
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)


print('데이터 로드 완료')

# # DataLoader의 구성 확인
# for i, batch in enumerate(dataloader):
#     print(f"Batch {i + 1}:")
#     print("Input shape:", batch['A'].shape) # [1, 3, 2453, 234]
#     print("Target shape:", batch['B'].shape) # [1, 3, 234, 253]
##################################

###### Testing######

# Create output dirs if they don't exist
if not os.path.exists('output/A'):
    os.makedirs('output/A')
if not os.path.exists('output/B'):
    os.makedirs('output/B')


for i, batch in enumerate(dataloader): # 10개의 batch
    
    # # Resize 변환을 사용하여 크기 조정
    # resize_transform = transforms.Compose([
    #     transforms.ToPILImage(),  # 텐서를 PIL 이미지로 변환
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor(),  # Tensor로 변환
    # ])

    # # 이미지 리사이징
    # batch['A'] = resize_transform(batch['A'])
    # batch['B'] = resize_transform(batch['B']) 
        
    print('A:', batch['A'].shape) # [1, 3, 256, 256] 나와야 함
    print('B:', batch['B'].shape)
    
    # Set model input
    # real_A = Variable(input_A.copy_(batch['A']))
    # real_B = Variable(input_B.copy_(batch['B']))
    
    real_A = batch['A'].to(device = 'cpu')
    # real_B = batch['B'].to(device = 'cpu')

    # Generate output
    fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
    # fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

    # Save image files
    # save_image(fake_A, 'output/A/%04d.png' % (i+1))
    save_image(fake_B, 'output/B/%04d.png' % (i+1))

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################
