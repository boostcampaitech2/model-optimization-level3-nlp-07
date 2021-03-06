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


## ๐ถ TEAM : ์กฐ์งKLUE๋
### ๐ Members  

๊น๋ณด์ฑ|๊น์งํ|๊นํ์|๋ฐ์ด์ญ|์ด๋ค๊ณค|์ ๋ฏธ์|์ ๋ํด
:-:|:-:|:-:|:-:|:-:|:-:|:-:
![image1][image1]|![image2][image2]|![image3][image3]|![image4][image4]|![image5][image5]|![image6][image6]|![image7][image7]
[Github](https://github.com/Barleysack)|[Github](https://github.com/JIHOO97)|[Github](https://github.com/vgptnv)|[Github](https://github.com/Tentoto)|[Github](https://github.com/DagonLee)|[Github](https://github.com/ekdub92)|[Github](https://github.com/Doohae)


### ๐ Contribution
`๊น๋ณด์ฑ` `DALI` , `Neural Architecture Search` , `Parameter Reduction`

`๊น์งํ` `Pruning`, `Quantization`, `fine-tuning`

`๊นํ์` `Neural Architecture Search`, `Stratified Subsampling`

`๋ฐ์ด์ญ` `Neural Architecture Search`, `Module Implementation`

`์ด๋ค๊ณค` `(Unable to participate in the competition)`

`์ ๋ฏธ์` `Neural Architecture Search`, `Hyperparameter Tuning`

`์ ๋ํด` `NAS`, `Data Augmentation`, `Baseline Modification`, `Hyperparameter Tuning`

[image1]: https://avatars.githubusercontent.com/u/56079922?v=4
[image2]: https://avatars.githubusercontent.com/u/57887761?v=4
[image3]: https://avatars.githubusercontent.com/u/62708568?v=4
[image4]: https://avatars.githubusercontent.com/u/80071163?v=4
[image5]: https://avatars.githubusercontent.com/u/43575986?v=4
[image6]: https://avatars.githubusercontent.com/u/42200769?v=4
[image7]: https://avatars.githubusercontent.com/u/80743307?v=4

<br/>


# 2. Project Outline
ํ๋ก์ ํธ ๊ธฐ๊ฐ : 2021.11.22 - 2021.12.02 (2 Weeks)

### Dataset

- ์ฌ์ฉ๋ ๋ฐ์ดํฐ์ : `TACO` (Segmentation / Object detection task๋ฅผ ํ๊ธฐ ์ํด ์ ์๋ COCO format์ ์ฌํ์ฉ ์ฐ๋ ๊ธฐ ๋ฐ์ดํฐ, CC-BY-4.0)
- ๋จ์ํ Classification ๋ฌธ์ ๋ก ์ค์ ํ๊ธฐ ์ํด TACO ๋ฐ์ดํฐ์์ Bounding box๋ฅผ cropํ ๋ฐ์ดํฐ๊ฐ ์ ๊ณต๋จ
- train data : ์ด 20,851 ์ฅ์ .jpg format ์ด๋ฏธ์ง
- test data : ์ด 5,217 ์ฅ์ .jpg format ์ด๋ฏธ์ง, private 2,611 ์ฅ, public 2,606 ์ฅ์ผ๋ก ๊ตฌ์ฑ๋จ
- train, validation, test data๋ ์ด 6๊ฐ์ ์นดํ๊ณ ๋ฆฌ (Metal, Paper, Paperpack, Plastic, Plasticbag, Styrofoam) ๋ก ์ด๋ฃจ์ด์ง

### Data Augmentation

- `optuna` ๋ชจ๋์ ํ์ฉํ Random Augmentation ์์์ ํตํด ์ฑ๋ฅ ํฅ์์ ์ด๋์ด๋ด๋ augmentation ๊ธฐ๋ฒ๋ค ์ ์ 
- ์ ์ ๋ ๊ธฐ๋ฒ๋ค์ ์ ์ฉํ์ฌ ์๋ณธ ์ด๋ฏธ์ง ๊ฐ์ 2๋ฐฐ์ ํด๋นํ๋ augmentation์ด ์ ์ฉ๋ ๋ฐ์ดํฐ๋ฅผ ์ถ๊ฐ๋ก ์ ์ฅ ํ ํ์ต โ ์ด๊ธฐ ์ด๋ฏธ์ง์ 3๋ฐฐ์ ํด๋นํ๋ ๊ฐ์์ ์ด๋ฏธ์ง ํ์ต
- ์ ์ฉ Augmentation ๊ธฐ๋ฒ
    
    `Invert` `Contrast` `AutoContrast` `Rotate` `TranslateX` `TranslateY` `Cutout` `Brightness` `Equalize` `Solarize` `Posterize` `Sharpness` `ShearX` `ShearY`
    

### Model Compression

- **Pruning**
    
    pre-trained model์ Unstructured Pruning์ ์ ์ฉํ์ฌ ๊ฐ์ค์น๊ฐ ๋ฎ์ neuron๋ค์ layer_type์ผ๋ก๋ถํฐ ์ ๊ฑฐํจ
    
- **Quantization**
    
    `Dynamic quantization`๊ณผ `static quantization`์ ์ด์ฉํ์ฌ ๋ฐ์ดํฐ๋ฅผ `int8 dtype`์ผ๋ก ๋ฐ๊ฟ
    

### Neural Architecture Search

- `Optuna`๋ฅผ ํ์ฉํด Neural Architecture Search ์คํ
- ํจ์จ์ ์ธ NAS ํ์ต์ ์ํด
    - ๋ฐ์ดํฐ์์ Subset์ Stratified ํ๊ฒ ์ถ์ถํ์ฌ ์ ์ ์์ ๋ฐ์ดํฐ์์ผ๋ก ํจ์จ์ ์ธ ๋ชจ๋ธ ํ์
    - 5-10 epochs๋ง ๊ฐ๋ณ๊ฒ ํ์ตํ์ฌ ๋ชจ๋ธ์ ๋ก์๋ง ํ์ธ!
- `mean_time`, `f1_score`, `flops`, `params_nums`๋ฅผ `Optuna` ์ objective ํ๊ฐ ์งํ๋ก ๋ฃ์ด์ฃผ์ด ๊ฒฝ๋ํ ๋ชจ๋ธ ํ์

### Evaluation

- ์ต์ข score๋ ๋ชจ๋ธ์ F1-score์ ๋ชจ๋ธ์ Submit time์ ํตํด ๊ณ์ฐ๋จ
- F1-score : ๋ถ๋ฅ ์ฑ๋ฅ ์งํ๋ก, ๊ธฐ์ค์ด ๋๋ ๋ชจ๋ธ์ F1 score์์ ์ ์ถํ ๋ชจ๋ธ์ F1 score์ ์ฐจ์ด๋ฅผ ๊ตฌํ ๋ค, ์์๋ฅผ ๊ณฑํ๊ณ  sigmoid ํจ์๋ฅผ ์ ์ฉํ ๊ฐ
- Submit time : ๊ธฐ์ค์ด ๋๋ ๋ชจ๋ธ์ ์ถ๋ก ํ๋ ์๊ฐ์ผ๋ก ์ ์ถํ ๋ชจ๋ธ์ ์ถ๋ก ํ๋ ์๊ฐ์ ๋๋ ๊ฐ

### Project Management

- ์ต์ข ํ๋ก์ ํธ์ ๋ณํํด์ ์งํ๋๊ธฐ ๋๋ฌธ์ ํจ์จ์ ์ธ ์๊ฐ ๋ถ๋ฐฐ๋ฅผ ์ํด ๋งค์ผ ์์นจ 10์์ Stand-Up Meeting์ ์งํ

### Final Score (Public)
| Metric | Score |
| --- | --- |
| `f1 score` | 0.6769 |
| `Inference time` | 57.0690 |





# 3. How to Use

## **Installation**

๋ค์๊ณผ ๊ฐ์ ๋ช๋ น์ด๋ก ํ์ํ libraries๋ฅผ ๋ค์ด ๋ฐ์ต๋๋ค.

`pip install -r requirements.txt`

## **Dataset**

๋๋ ํ ๋ฆฌ : data

(๋ณธ Repository์ data๋ ํฌํจ๋์ด ์์ง ์์ต๋๋ค.)

## **Configs for data and model**

๋๋ ํ ๋ฆฌ : code/configs

## **Model Optimization**

ํ์ผ : tune.py, train.py, inference.py

## **Modeling**

๋๋ ํ ๋ฆฌ : code/src


- `code` ํด๋ ์์๋ optuna๋ฅผ ํตํด ์ต์ ํ ๋ชจ๋ธ์ ์ฐพ๋ ํ์ผ tune.py ํ์ผ์ด ๋ค์ด์์ต๋๋ค. 
- `code` ํด๋ ์์๋ TACO dataset์ tune.py๋ฅผ ํตํด ์ฐพ์ ๋ชจ๋ธ์ ํตํด ํ์ต๊ณผ ์ถ๋ก ์ ์คํ์ํฌ ์ ์๋ train.py ํ์ผ๊ณผ inference.py ํ์ผ์ด ๋ค์ด์์ต๋๋ค.
