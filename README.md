# Shape Robust Text Detection with Progressive Scale Expansion Network

## Update 
### 20190401

1. add author loss, the results are compared in [Performance](#Performance)


| Method                   | Precision (%) | Recall (%) | F-measure (%) | FPS(1080Ti) |
|--------------------------|---------------|------------|---------------|-----|
| PSENet-1s with resnet152 batch 8 myloss | 85.47         | 78.76      | 81.98         | 1.49 |
| PSENet-1s with resnet152 batch 8 author loss| 85.04         | 79.68     | 82.27         | 1.48 |

### Download
resnet50 and resnet152: [bauduyun](https://pan.baidu.com/s/1rN0oGBRsdUYmcQUayMZUOA) extract code: rxjf

## Train
1. config the `trainroot`,`testroot`in [config.py](config.py)
2. use fellow script to run
```sh
python3 train.py
```

## Test
[eval.py](eval.py) is used to test model on test dataset

1. config `model_path`, `data_path`, `gt_path`, `save_path` in [eval.py](eval.py)
2. use fellow script to test
```sh
python3 eval.py
```

## Predict 
[predict.py](predict.py) is used to inference on single image

1. config `model_path`, `img_path`, `gt_path`, `save_path` in [predict.py](predict.py)
2. use fellow script to predict
```sh
python3 predict.py
```

The project is still under development.

<h2 id="Performance">Performance</h2>

### [ICDAR 2015](http://rrc.cvc.uab.es/?ch=4&com=evaluation&task=1)
only train on ICDAR2015 dataset with single NVIDIA 1080Ti

my implementation with my loss use adam and warm_up

| Method                   | Precision (%) | Recall (%) | F-measure (%) | FPS(1080Ti) |
|--------------------------|---------------|------------|---------------|-----|
| PSENet-1s with resnet50 batch 8  | 80.90 | 77.32 | 79.07 | 1.76 |
| PSENet-2s with resnet50 batch 8  | 80.86 | 77.42 | 79.05 | 3.55 |
| PSENet-4s with resnet50 batch 8  | 80.09 | 76.70 | 78.36 | 4.43 |
| PSENet-1s with resnet152 batch 4 | 85.24 | 80.11 | 82.60 | 1.48 |
| PSENet-2s with resnet152 batch 4 | 85.02 | 80.06 | 82.46 | 2.56 |
| PSENet-4s with resnet152 batch 4 | 83.46 | 79.00 | 81.17 | 2.99 |

my implementation with author loss use adam and warm_up

| Method                   | Precision (%) | Recall (%) | F-measure (%) | FPS(1080Ti) |
|--------------------------|---------------|------------|---------------|-----|
| PSENet-1s with resnet50 batch 8  | tbd | tbd | tbd | 1.76 |
| PSENet-2s with resnet50 batch 8  | tbd | tbd | tbd | 3.55 |
| PSENet-4s with resnet50 batch 8  | tbd | tbd | tbd | 4.43 |
| PSENet-1s with resnet152 batch 4 | tbd | tbd | tbd | 1.48 |
| PSENet-2s with resnet152 batch 4 | tbd | tbd | tbd | 2.56 |
| PSENet-4s with resnet152 batch 4 | tbd | tbd | tbd | 2.99 |

my implementation with author loss use adam and MultiStepLR

| Method                   | Precision (%) | Recall (%) | F-measure (%) | FPS(1080Ti) |
|--------------------------|---------------|------------|---------------|-----|
| PSENet-1s with resnet50 batch 8  | 83.49 | 79.62 | 81.51 | 1.76 |
| PSENet-2s with resnet50 batch 8  | 83.37 | 79.68 | 81.48 | 3.55 |
| PSENet-4s with resnet50 batch 8  | 82.44 | 78.91 | 80.63 | 4.43 |
| PSENet-1s with resnet152 batch 4 | 85.04 | 79.68 | 82.27 | 1.48 |
| PSENet-2s with resnet152 batch 4 | 84.88 | 79.20 | 81.94 | 2.56 |
| PSENet-4s with resnet152 batch 4 | 83.81 | 78.76 | 81.21 | 2.99 |

official implementation use SGD and StepLR

| Method                   | Precision (%) | Recall (%) | F-measure (%) | FPS(1080Ti) |
|--------------------------|---------------|------------|---------------|-----|
| PSENet-1s with resnet50 batch 8  |  84.15 | 80.26  | 82.16 | 1.76 |
| PSENet-2s with resnet50 batch 8  |  83.61 | 79.82  | 81.67 | 3.72 |
| PSENet-4s with resnet50 batch 8  |  81.90 | 78.23  | 80.03 | 4.51 |
| PSENet-1s with resnet152 batch 4 |  82.87 | 78.76  | 80.77 | 1.53 |
| PSENet-2s with resnet152 batch 4 |  82.33 | 78.33  | 80.28 | 2.61 |
| PSENet-4s with resnet152 batch 4 |  81.19 | 77.13  | 79.11 | 3.00 |

### examples
![](imgs/img_31.jpg)

![](imgs/img_73.jpg)

![](imgs/img_83.jpg)

![](imgs/img_98.jpg)

![](imgs/img_125.jpg)

### reference
1. https://github.com/liuheng92/tensorflow_PSENet
2. https://github.com/whai362/PSENet