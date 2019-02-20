# Shape Robust Text Detection with Progressive Scale Expansion Network

## Introduction
Progressive Scale Expansion Network (PSENet) is a text detector which is able to well detect the arbitrary-shape text in natural scene.

![Figure 1: Illustration of our overall pipeline.](imgs//pipeline.png)

![Figure 2: The procedure of progressive scale expansion algorithm.](imgs/pse.png)

# train
1. config the `trainroot`,`testroot`in [config.py](config.py)
2. use fellow script to run
```sh
python3 train.py
```

# test
[eval.py](eval.py) is used to test model on test dataset

1. config `model_path`, `data_path`, `gt_path`, `save_path` in [eval.py](eval.py)
2. use fellow script to test
```sh
python3 eval.py
```

# predict 
[predict.py](predict.py) is used to inference on single image

1. config `model_path`, `img_path`, `gt_path`, `save_path` in [predict.py](predict.py)
2. use fellow script to predict
```sh
python3 predict.py
```

The project is still under development.

# Performance
### [ICDAR 2015](http://rrc.cvc.uab.es/?ch=4&com=evaluation&task=1)
| Method                   | Precision (%) | Recall (%) | F-measure (%) | fps |
|--------------------------|---------------|------------|---------------|-----|
| PSENet-1s with resnet152 | 76.46         | 74.77      | 75.60         | 1.4 |