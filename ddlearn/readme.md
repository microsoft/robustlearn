# Generalizable Low-Resource Activity Recognition with Diverse and Discriminative Representation Learning

This project implements our paper [Generalizable Low-Resource Activity Recognition with Diverse and Discriminative Representation Learning](https://arxiv.org/abs/2306.04641). Please refer to our paper [1] for the method and technical details. 

![](resources/figures/fig-framework.pdf)


**Abstract:** Human activity recognition (HAR) is a time series classification task that focuses on identifying the motion patterns from human sensor readings. Adequate data is essential but a major bottleneck for training a generalizable HAR model, which assists customization and optimization of online web applications. However, it is costly in time and economy to collect large-scale labeled data in reality, i.e., the low-resource challenge. Meanwhile, data collected from different persons have distribution shifts due to different living habits, body shapes, age groups, etc. The low-resource and distribution shift challenges are detrimental to HAR when applying the trained model to new unseen subjects. In this paper, we propose a novel approach called Diverse and Discriminative representation Learning (DDLearn) for generalizable low-resource HAR. DDLearn simultaneously considers diversity and discrimination learning. With the constructed self-supervised learning task, DDLearn enlarges the data diversity and explores the latent activity properties. Then, we propose a diversity preservation module to preserve the diversity of learned features by enlarging the distribution divergence between the original and augmented domains. Meanwhile, DDLearn also enhances semantic discrimination by learning discriminative representations with supervised contrastive learning. Extensive experiments on three public HAR datasets demonstrate that our method significantly outperforms state-of-art methods by an average accuracy improvement of 9.5% under the low-resource distribution shift scenarios, while being a generic, explainable, and flexible framework.


## Requirement

The required packages are listed in `requirements.txt` for minimum requirement (Python 3.8.5):

```
$ pip install -r requirements.txt
$ pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

## Dataset 
DSADS. UCI Daily and Sports Data Set collects data from 8 subjects around 1.14M samples. Three kinds of body-worn sensor units including triaxial accelerometer, triaxial gyroscope, and triaxial magnetometer are worn on 5 body positions of each subject: torso, right arm, left arm, right leg, and left leg. It consists of 19 activities. The total signal duration is 5 minutes for each activity of each subject.

In the following links are data(100% propotion of training data with seed=1 as examples, you can random sample different propotion from the training data to simulate the low-resource regime) after preprocessing according to the paper. You can get different propotion of training data with the 100% data. 

All data preprocess code are provided, you can process the data of the other two datasets with different propotions of training data using these codes.

```
wget https://wjdcloud.blob.core.windows.net/dataset/kdd-act-dsads/dsads_crosssubject_rawaug_rate1.0_t0_seed1_scalerminmax.pkl 
wget https://wjdcloud.blob.core.windows.net/dataset/kdd-act-dsads/dsads_crosssubject_rawaug_rate1.0_t1_seed1_scalerminmax.pkl 
wget https://wjdcloud.blob.core.windows.net/dataset/kdd-act-dsads/dsads_crosssubject_rawaug_rate1.0_t2_seed1_scalerminmax.pkl 
wget https://wjdcloud.blob.core.windows.net/dataset/kdd-act-dsads/dsads_crosssubject_rawaug_rate1.0_t3_seed1_scalerminmax.pkl 
```

## How to run

We provide the commands for 26 tasks (100% and 20% training data) in DSADS, PAMAP2 and USC-HAD datasets to reproduce the results, change the root_path and save_path as your path.

```
sh script.sh
```

## Results

## Results

**DSADS 100% training data** 

| Source   | 1,2,3      | 0,2,3      | 0,1,3      | 0,1,2      | AVG        |
|----------|------------|------------|------------|------------|------------|
| Target   | 0          | 1          | 2          | 3          | -          |
| ERM      | 63.28      | 60.94      | 64.40      | 71.54      | 65.04      |
| Mixup    | 84.26      | 82.48      | 85.16      | 81.25      | 83.29      |
| Mldg     | 72.99      | 69.53      | 73.21      | 72.32      | 72.01      |
| RSC      | 65.40      | 80.25      | 76.79      | 75.11      | 74.39      |
| AND-Mask | 78.35      | 70.76      | 84.15      | 65.51      | 74.69      |
| SimCLR   | 72.54      | 75.65      | 77.08      | 73.83      | 74.78      |
| Fish     | 62.05      | 65.85      | 77.01      | 74.22      | 69.78      |
| DDLearn  | **93.81**  | **93.15**  | **92.49**  | **88.62**  | **92.02**  |

**DSADS 20% training data**

| Source   | 1,2,3      | 0,2,3      | 0,1,3      | 0,1,2      | AVG        |
|----------|------------|------------|------------|------------|------------|
| Target   | 0          | 1          | 2          | 3          | -          |
| ERM      | 56.92      | 68.30      | 58.93      | 73.88      | 64.51      |
| Mixup    | 75.22      | 71.32      | 70.20      | 70.98      | 71.93      |
| Mldg     | 59.60      | 69.20      | 68.08      | 69.42      | 66.58      |
| RSC      | 53.24      | 65.40      | 67.97      | 63.17      | 62.45      |
| AND-Mask | 58.59      | 63.06      | 70.09      | 62.05      | 63.45      |
| SimCLR   | 69.01      | 75.13      | 79.02      | 77.34      | 75.13      |
| Fish     | 53.01      | 62.95      | 69.75      | 66.52      | 63.06      |
| DDLearn  | **90.50**  | **90.28**  | **90.83**  | **84.53**  | **89.04**  |

**PAMAP2 100 training data**

| Source   | 1,2,3      | 0,2,3      | 0,1,3      | 0,1,2      | AVG        |
|----------|------------|------------|------------|------------|------------|
| Target   | 0          | 1          | 2          | 3          | -          |
| ERM      | 49.61      | 90.63      | 83.59      | 72.27      | 74.02      |
| Mixup    | 69.53      | 91.41      | 89.84      | 81.25      | 83.01      |
| Mldg     | 57.03      | 90.62      | 87.89      | 75.78      | 77.83      |
| RSC      | 52.34      | 89.84      | 87.89      | 80.47      | 77.64      |
| AND-Mask | 55.86      | 89.84      | 85.94      | 78.12      | 77.44      |
| SimCLR   | 62.89      | 83.59      | 80.86      | 74.61      | 75.49      |
| Fish     | 54.30      | 92.97      | 90.62      | 76.17      | 78.52      |
| DDLearn  | **81.02**  | **94.85**  | **88.34**  | **88.77**  | **88.25**  |

**PAMAP2 20 training data**

| Source   | 1,2,3      | 0,2,3      | 0,1,3      | 0,1,2      | AVG        |
|----------|------------|------------|------------|------------|------------|
| Target   | 0          | 1          | 2          | 3          | -          |
| ERM      | 47.66      | 74.61      | 78.91      | 78.13      | 69.82      |
| Mixup    | 57.03      | 76.17      | 82.03      | 76.56      | 72.95      |
| Mldg     | 52.34      | 78.52      | 78.52      | 75.39      | 71.19      |
| RSC      | 55.08      | 84.38      | 83.59      | 76.59      | 75.00      |
| AND-Mask | 51.56      | 85.16      | 83.59      | 75.00      | 73.83      |
| SimCLR   | 61.33      | 83.20      | 81.25      | 73.83      | 74.90      |
| Fish     | 48.83      | 87.11      | 84.38      | 73.05      | 73.34      |
| DDLearn  | **74.82**  | **87.13**  | **86.57**  | **82.97**  | **82.87**  |


**USC-HAD 100% training data**

| Source   | 1,2,3,4    | 0,2,3,4    | 0,1,3,4    | 0,1,2,4    | 0,1,2,3    | AVG        |
|----------|------------|------------|------------|------------|------------|------------|
| Target   | 0          | 1          | 2          | 3          | 4          |-           |
| ERM      | 83.07      | 74.74      | 79.17      | 53.39      | 64.58      | 70.99      |
| Mixup    | 84.11      | 81.25      | 86.20      | 72.66      | 75.00      | 79.84      |
| Mldg     | 79.17      | 71.09      | 77.08      | 65.62      | 60.68      | 70.73      |
| RSC      | 83.59      | 77.86      | 84.38      | 70.31      | 65.62      | 76.35      |
| AND-Mask | 82.29      | 76.56      | 80.73      | 63.02      | 69.53      | 74.43      |
| SimCLR   | 76.17      | 75.00      | 82.03      | 62.76      | 65.89      | 72.37      |
| Fish     | 83.85      | 77.08      | 86.72      | 65.36      | 74.22      | 77.45      |
| DDLearn  | **86.09**  | **83.33**  | **88.38**  | **79.23**  | **76.96**  | **82.80**  |

**USC-HAD 20% training data**

| Source   | 1,2,3,4    | 0,2,3,4    | 0,1,3,4    | 0,1,2,4    | 0,1,2,3    | AVG        |
|----------|------------|------------|------------|------------|------------|------------|
| Target   | 0          | 1          | 2          | 3          | 4          |-           |
| ERM      | 67.97      | 66.93      | 66.93      | 55.21      | 59.64      | 63.33      |
| Mixup    | 65.62      | 70.57      | 72.66      | 57.55      | 61.20      | 65.52      |
| Mldg     | 67.19      | 61.46      | 62.24      | 57.03      | 59.64      | 61.51      |
| RSC      | 74.22      | 72.14      | 75.52      | 58.33      | 64.58      | 68.96      |
| AND-Mask | 65.36      | 69.53      | 71.88      | 58.59      | 57.29      | 64.53      |
| SimCLR   | 67.97      | 68.36      | 78.13      | 59.90      | 61.72      | 67.21      |
| Fish     | 72.92      | 73.44      | 67.71      | 53.65      | 63.02      | 66.15      |
| DDLearn  | **77.22**  | **78.68**  | **80.56**  | **71.01**  | **78.23**  | **77.14**  |


## Contact

- qinxin@ict.ac.cn
- jindongwang@outlook.com

## References

```
@article{qin2023generalizable,
  title={Generalizable Low-Resource Activity Recognition with Diverse and Discriminative Representation Learning},
  author={Qin, Xin and Wang, Jindong and Ma, Shuo and Lu, Wang and Zhu, Yongchun and Xie, Xing and Chen, Yiqiang},
  journal={KDD},
  year={2023}
}
```