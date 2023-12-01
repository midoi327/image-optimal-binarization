# image-binarization-algorithm
이미지 전처리 품질지표(PSNR, SSIM) 최적화 알고리즘

![Untitled (2)](https://github.com/midoi327/image-preprocessing-algorithm/assets/50612011/1ac868cf-7766-4c9c-98f4-715c31c92824)

---
## 📌 development environment
```Ubuntu 18.04```

```Python 3.9.5```

---


## 📌 pseudo code
### forloop.py
```python
  For each brightness in range(-100, 100, 10): // brightness for loop
      For each contrast in range(1, 2.1, 0.5): // contrast for loop
          For each threshold in range(50, 200, 10): // binarization threshold for loop
              For each morph in range(1, 10, 2): // morphing threshold for loop
                  Try:
                      Enhance input_image with current parameters
                      Get psnr_value, ssim_value by evaluating quality against target image

                      If psnr_value > best_psnr and ssim_value > best_ssim:
                          Increment count
                          Update best_psnr, best_ssim, best_params
                          Print result and save enhanced_image

```


---

## 📌 Quick Start

1. **input folder** 생성 후 원본 이미지 넣어두기
2. **target folder** 생성 후 원본 이미지를 가장 잘 이진화한 이미지 넣어두기 (psnr, ssim 점수 계산을 위해)
3. **output folder** 생성 ( 결과 이미지 저장 )
4.  ``` pip install -r requirements.txt ```
5.  ``` python forloop.py ```
6. **output folder** 확인

---



## 📌 Develop
![image](https://github.com/midoi327/image-preprocessing-algorithm/assets/50612011/904ff269-48a4-4160-bd20-32c15a4fd49e)


---

### 📌 Contact

If you have any questions, please contact ```midoi327@naver.com```
