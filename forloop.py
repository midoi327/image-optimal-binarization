import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
from piq import ssim
import torch
import torch.nn as nn
import os
from tqdm import tqdm


class BestImage:
    def __init__(self, input_folder, target_folder, output_folder):
        
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.output_folder = output_folder
        self.invert_counts = 0
        self.best_psnr = 15
        self.best_ssim = 0.5
        self.best_params = {}
        self.input_image = None

    def display_image(self, image):
        if image is not None:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                plt.imshow(image)
            else:
                plt.imshow(image, cmap='gray')
        else:
            print('이미지가 존재하지 않습니다.')

    def save_image(self, image, filename='new_image', count=0):
        if image is not None:
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

            cv2.imwrite(f"./output/{filename}_{count}.png", gray_image)
            # print(f'Saved...{filename}')

    def invert_colors(self, image):
        self.invert_counts += 1
        inverted = 255 - image
        new_image = inverted + self.invert_counts * image // (self.invert_counts + 1)
        return new_image

    def adjust_brightness(self, img, brightness_factor=30):
        adjusted_image = cv2.convertScaleAbs(img, alpha=1, beta=brightness_factor)
        return adjusted_image

    def adjust_contrast(self, img, contrast_factor=1.5):
        adjusted_image = cv2.convertScaleAbs(img, alpha=contrast_factor, beta=0)
        return adjusted_image

    def convert_to_binary(self, img, threshold):
        _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        return binary_img

    def remove_small_noise(self, binary_image, diameter_threshold_mm=8):
        kernel_size = diameter_threshold_mm
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        result_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        return result_image

    def grayscale_image(self, image):
        is_gray = len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1)
        if is_gray:
            image = image
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def psnr(self, input, target, data_range=1.0):
        mse = torch.mean((input - target) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * torch.log10((data_range ** 2) / mse)

    def psnr_ssim(self, image, target_file):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        target_image = Image.open(target_file).convert("RGB")

        preprocess = transforms.Compose([transforms.ToTensor()])
        transform = transforms.Compose([transforms.Resize((image.shape[0], image.shape[1]))])

        image = preprocess(image)
        target_image = preprocess(transform(target_image))
        image = image.unsqueeze(0)
        target_image = target_image.unsqueeze(0)

        psnr_value = self.psnr(image, target_image, data_range=1.0)
        ssim_value = ssim(image, target_image, data_range=1.0)
        return psnr_value, ssim_value

    def enhance_image(self, brightness, threshold, contrast, morph):
        enhanced_image = self.adjust_brightness(self.input_image, brightness_factor=brightness)
        enhanced_image = self.adjust_contrast(enhanced_image, contrast_factor=contrast)
        enhanced_image = self.grayscale_image(enhanced_image)
        enhanced_image = self.convert_to_binary(enhanced_image, threshold=threshold)
        enhanced_image = self.remove_small_noise(enhanced_image, diameter_threshold_mm=morph)
        return enhanced_image

    def find_best_image(self):
        
        # 폴더 내 이미지 선택
        image_files = sorted(os.listdir(self.input_folder))
        print(f'Collected {len(image_files)} Data..\n')
        
        for f in image_files:

            self.input_image = cv2.imread(os.path.join(self.input_folder, f))
            print(f'Input Image is {f}. Processing...')
            
            inv = int(input('Input Image 색 반전이 필요하면 1을 입력하세요. 필요없다면 0을 입력하세요.'))
            if inv == 1:
                self.input_image = self.invert_colors(self.input_image) 
            
            self.best_psnr = float(input('PSNR 기준 점수를 입력하세요 : '))
            self.best_ssim = float(input('SSIM 기준 점수를 입력하세요 : '))
            
            count = 0 # 베스트 이미지가 몇 번 갱신되었는가 
            self.best_params = {} # 파라미터 초기화
            
            for brightness in tqdm(range(-100, 100, 10), desc=f'Collecting Data', leave=True): # 밝기조절
                for contrast in np.arange(1, 2.1, 0.5): # 대비조절
                    for threshold in tqdm(range(50, 200, 10), desc='이진화', leave=False): # 이진화 조절
                        for morph in np.arange(3, 10, 2): # 모폴로지 조절
                            try:
                                enhanced_image = self.enhance_image(brightness, threshold, contrast, morph)
                                psnr_value, ssim_value = self.psnr_ssim(enhanced_image, os.path.join(self.target_folder, f)) # ////////////////////target 이미지 
                                
                                if psnr_value > self.best_psnr and ssim_value > self.best_ssim : 
                                    count += 1
                                    
                                    self.best_psnr = psnr_value 
                                    self.best_ssim = ssim_value
                                    self.best_params = {'brightness': brightness, 'threshold': threshold, 'contrast': contrast, 'morph': morph}
                                    
                                    print(f'**{count}번째 이미지 psnr :{self.best_psnr:.3f}, ssim: {self.best_ssim:.3f} with {self.best_params}')
                                    self.save_image(image=enhanced_image, filename= f.replace('.png', ''), count = count)
                                    
                            except Exception as e:
                                print(f'Exception occurred: {e}')

            print(f'{f} result: best_psnr={self.best_psnr:.3f}, best_ssim: {self.best_ssim:.3f}, best_params={self.best_params}\n\n')



input_folder = './input'
target_folder = './target'
output_folder = './output'


best_instance = BestImage(input_folder, target_folder, output_folder)
best_instance.find_best_image()
