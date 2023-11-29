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
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import display
import graphviz

class BestImage:
    def __init__(self, input_folder, target_folder, output_folder):
        
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.output_folder = output_folder
        self.invert_counts = 0
        self.best_psnr = 10
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


    def visualize_decision_tree(self, tree, feature_names):
        dot_data = export_graphviz(tree, out_file=None,
                                   filled=True, rounded=True,
                                   special_characters=True, feature_names=feature_names)
        graph = graphviz.Source(dot_data)
        graph.render('decision_tree', format='png', cleanup=False)
        # display(Image.open('decision_tree.png'))
        
    def decision_tree(self):
        
        # 폴더 내 첫 번째 이미지 선택
        image_files = os.listdir(self.input_folder)
        self.input_image = cv2.imread(os.path.join(self.input_folder, image_files[0]))
        print(f'Input Image is {image_files[0]}. Processing Data...')

        self.input_image = self.invert_colors(self.input_image) # //////////////////////////////////////////////이미지에 따라 선택 적용

        # 특징벡터 생성
        X = []
        y_psnr = []
        y_ssim = []
        point = []

        for brightness in tqdm(range(-100, 100, 10), desc=f'Collecting Data', leave=True):
            for contrast in np.arange(1, 2.1, 0.5):
                for threshold in tqdm(range(50, 200, 10), desc='이진화', leave=False):
                    for morph in np.arange(1, 10, 2):
                        enhanced_image = self.enhance_image(brightness, threshold, contrast, morph)
                        psnr_value, ssim_value = self.psnr_ssim(enhanced_image, '/home/piai/문서/miryeong/Algorithm_1/target/saved_image2.png')

                        X.append([brightness, threshold, contrast, morph])
                        y_psnr.append(psnr_value.item())
                        y_ssim.append(ssim_value.item())
                        point.append(0.7*psnr_value.item() + 0.3*ssim_value.item())
                            
                            
                                
        # 결정나무 훈련
        tree = DecisionTreeRegressor()
        
        X = np.array(X)
        point = np.array(point)
        
        tree.fit(X, point)

        # 베스트 파라미터 찾기
        
        print(f'Collected {len(X)} Data...\n')
        print('predict value of ''tree''')
        print(tree.predict(X))
        print()
        
        print(f'best point : {np.max(tree.predict(X)):.3f}')
        
        best_index = np.argmax(tree.predict(X))

        self.best_params['brightness'] = X[best_index, 0]
        self.best_params['threshold'] = X[best_index, 1]
        self.best_params['contrast'] = X[best_index, 2]
        self.best_params['morph'] = X[best_index, 3]


        print(f'Best parameters result: {self.best_params}')
        
        
        # 결정 트리 시각화
        feature_names = ['brightness', 'threshold', 'contrast', 'morph']

        self.visualize_decision_tree(tree, feature_names)

        # 베스트 결과 이미지 저장
        best_image = self.enhance_image(brightness=self.best_params['brightness'], threshold=self.best_params['threshold'], contrast=self.best_params['contrast'], morph=int(self.best_params['morph']))
 
        self.save_image(image = best_image, filename='best_image', count=1)






# 예제 사용
input_folder = './input'
target_folder = './target'
output_folder = './output'

best_instance = BestImage(input_folder, target_folder, output_folder)
best_instance.decision_tree()