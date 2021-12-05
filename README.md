# Boostcamp Model Optimization Competition
## **Table of contents**

1. [Introduction](#1-introduction)
2. [Project Outline](#2-project-outline)
3. [How to Use](#3-how-to-use)


# 1. Introduction  
<br/>
<p align="center">
   <img src="https://user-images.githubusercontent.com/62708568/140640556-b2a0406c-09cd-48be-ae37-4c65e206b693.JPG" style="width:1000px;"/>
</p>

<br/>


## 🐶 TEAM : 조지KLUE니
### 🔅 Members  

김보성|김지후|김혜수|박이삭|이다곤|전미원|정두해
:-:|:-:|:-:|:-:|:-:|:-:|:-:
![image1][image1]|![image2][image2]|![image3][image3]|![image4][image4]|![image5][image5]|![image6][image6]|![image7][image7]
[Github](https://github.com/Barleysack)|[Github](https://github.com/JIHOO97)|[Github](https://github.com/vgptnv)|[Github](https://github.com/Tentoto)|[Github](https://github.com/DagonLee)|[Github](https://github.com/ekdub92)|[Github](https://github.com/Doohae)


### 🔅 Contribution
`김보성` `DALI` , `Neural Architecture Search` , `Parameter Reduction`

`김지후` `Pruning`, `Quantization`, `fine-tuning`

`김혜수` `Neural Architecture Search`, `Stratified Subsampling`

`박이삭` `Neural Architecture Search`, `Module Implementation`

`이다곤` `(Unable to participate in the competition)`

`전미원` `Neural Architecture Search`, `Hyperparameter Tuning`

`정두해` `NAS`, `Data Augmentation`, `Baseline Modification`, `Hyperparameter Tuning`

[image1]: https://avatars.githubusercontent.com/u/56079922?v=4
[image2]: https://avatars.githubusercontent.com/u/57887761?v=4
[image3]: https://avatars.githubusercontent.com/u/62708568?v=4
[image4]: https://avatars.githubusercontent.com/u/80071163?v=4
[image5]: https://avatars.githubusercontent.com/u/43575986?v=4
[image6]: https://avatars.githubusercontent.com/u/42200769?v=4
[image7]: https://avatars.githubusercontent.com/u/80743307?v=4

<br/>


# 2. Project Outline
프로젝트 기간 : 2021.11.22 - 2021.12.02 (2 Weeks)

### Dataset

- 사용된 데이터셋 : `TACO` (Segmentation / Object detection task를 풀기 위해 제작된 COCO format의 재활용 쓰레기 데이터, CC-BY-4.0)
- 단순한 Classification 문제로 설정하기 위해 TACO 데이터셋의 Bounding box를 crop한 데이터가 제공됨
- train data : 총 20,851 장의 .jpg format 이미지
- test data : 총 5,217 장의 .jpg format 이미지, private 2,611 장, public 2,606 장으로 구성됨
- train, validation, test data는 총 6개의 카테고리 (Metal, Paper, Paperpack, Plastic, Plasticbag, Styrofoam) 로 이루어짐

### Data Augmentation

- `optuna` 모듈을 활용한 Random Augmentation 작업을 통해 성능 향상을 이끌어내는 augmentation 기법들 선정
- 선정된 기법들을 적용하여 원본 이미지 개수 2배에 해당하는 augmentation이 적용된 데이터를 추가로 저장 후 학습 ⇒ 초기 이미지의 3배에 해당하는 개수의 이미지 학습
- 적용 Augmentation 기법
    
    `Invert` `Contrast` `AutoContrast` `Rotate` `TranslateX` `TranslateY` `Cutout` `Brightness` `Equalize` `Solarize` `Posterize` `Sharpness` `ShearX` `ShearY`
    

### Model Compression

- **Pruning**
    
    pre-trained model에 Unstructured Pruning을 적용하여 가중치가 낮은 neuron들을 layer_type으로부터 제거함
    
- **Quantization**
    
    `Dynamic quantization`과 `static quantization`을 이용하여 데이터를 `int8 dtype`으로 바꿈
    

### Neural Architecture Search

- `Optuna`를 활용해 Neural Architecture Search 실험
- 효율적인 NAS 학습을 위해
    - 데이터셋의 Subset을 Stratified 하게 추출하여 적은 양의 데이터셋으로 효율적인 모델 탐색
    - 5-10 epochs만 가볍게 학습하여 모델의 떡잎만 확인!
- `mean_time`, `f1_score`, `flops`, `params_nums`를 `Optuna` 의 objective 평가 지표로 넣어주어 경량화 모델 탐색

### Evaluation

- 최종 score는 모델의 F1-score와 모델의 Submit time을 통해 계산됨
- F1-score : 분류 성능 지표로, 기준이 되는 모델의 F1 score에서 제출한 모델의 F1 score의 차이를 구한 뒤, 상수를 곱하고 sigmoid 함수를 적용한 값
- Submit time : 기준이 되는 모델의 추론하는 시간으로 제출한 모델의 추론하는 시간을 나눈 값

### Project Management

- 최종 프로젝트와 병행해서 진행됐기 때문에 효율적인 시간 분배를 위해 매일 아침 10시에 Stand-Up Meeting을 진행

### Final Score (Public)

   `f1 score` : 0.6769  
   `Inference time` : 57.0690   





# 3. How to Use

## **Installation**

다음과 같은 명령어로 필요한 libraries를 다운 받습니다.

`pip install -r requirements.txt`

## **Dataset**

디렉토리 : data

## **Configs for data and model**

디렉토리 : code/configs

## **Model Optimization**

파일 : tune.py, train.py, inference.py

## **Modeling**

디렉토리 : code/src


- `code` 폴더 안에는 optuna를 통해 최적화 모델을 찾는 파일 tune.py 파일이 들어있습니다. 
- `code` 폴더 안에는 TACO dataset을 tune.py를 통해 찾은 모델을 통해 학습과 추론을 실행시킬 수 있는 train.py 파일과 inference.py 파일이 들어있습니다.
