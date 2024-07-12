# SpecFormer: Guarding Vision Transformer Robustness via Maximum Singular Value Penalization

This is the official code for SpecFormer: Guarding Vision Transformer Robustness via Maximum Singular Value Penalization (ECCV 2024).


## Dependencies
- Python 3.8.17
- torch 1.9.0
- torchvision 0.10.0
- timm 0.5.4  
Run `pip install -r requirement.txt` to install all requrements.


## Directories

- `auto_LiRPA`: Contains the logger and `MultiAverageMeter`.
- `model_for_cifar`: Vanilla ViT variant models for CIFAR-10 and CIFAR-100 experiments.
- `model_for_cifar_sn`: SpecFormer models for CIFAR-10 and CIFAR-100 experiments.
- `model_for_imagenet`: Vanilla ViT variant models for ImageNet and Imagenette experiments.
- `model_for_imagenet_sn`: SpecFormer models for ImageNet and Imagenette experiments.
- `parser`: Python scripts for retrieving input parameters from the command line.
  - `parser_cifar.py`: Parser for CIFAR experiments.
  - `parser_imagenet.py`: Parser for ImageNet experiments.
  - `parser_imagenette.py`: Parser for Imagenette experiments.
- `robust_evaluate`: Python scripts for evaluating robustness.
  - `aa.py`: Evaluates AutoAttack.
  - `fgsm.py`: Evaluates FGSM Attack.
  - `pgd.py`: Evaluates PGD Attack.
- `train`: Python scripts for training models.
  - `train_cifar.py`: Training script for CIFAR experiments.
  - `train_imagenet.py`: Training script for ImageNet experiments.
  - `train_imagenette.py`: Training script for Imagenette experiments.
  - `utils.py`: Contains the data loading code.

这样描述更加具体，并且语法准确。
## Data

- **CIFAR-10 and CIFAR-100**: These datasets will be automatically downloaded when running `train_cifar` using `datasets.CIFAR10(args.data_dir, train=True, transform=train_transform, download=True)`.
- **ImageNet**: The ImageNet dataset can be downloaded from [ImageNet](https://www.image-net.org/download.php).
- **Imagenette-v1**: The Imagenette-v1 dataset can be downloaded from [Imagenette-v1](https://s3.amazonaws.com/fast-ai-imageclas/imagenette.tgz).


## Running

### CIFAR-10/100
```python
CUDA_VISIBLE_DEVICES=0 python -m train.train_cifar --model "vit_small_patch16_224_sn" --dataset cifar10 --out-dir "/log/" --method 'CLEAN'  --seed 0 --epochs 40 --data-dir /data/cifar --pen-for-qkv 1e-5 1e-5 1e-5  

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m train.train_cifar --model "vit_base_patch16_224_sn" --dataset cifar100 --out-dir "/log/" --method 'AT'  --seed 0 --epochs 40 --data-dir /data/cifar --pen-for-qkv 1e-5 1e-5 1e-5
```

You can switch to other ViT variants using the `--model` option, change the dataset with the `--dataset` option, and select a different method with the `--method` option. Additionally, you can adjust the penalizing strength using the `--pen-for-qkv` option, where the first value penalizes the query matrix, the second value penalizes the key matrices, and the third value penalizes the value matrices.

### ImageNet
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m train.train_imagenet --model "vit_base_patch16_224_in21k_sn"  --batch-size-eval 128 --AA-batch 128 --out-dir "/log/" --method 'CLEAN' --seed 0 --data-dir /data/imagenet/ImageNet/ --pen-for-qkv 5e-3 6e-4 7e-5  

```

### ImageNette
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m train.train_imagenette --model "deit_small_patch16_224_sn"  --out-dir "/log/" --method 'CLEAN'  --seed 0 --epochs 40 --data-dir /data/imagenette/ --pen-for-qkv 1e-5 1e-5 1e-5
```


## Acknowlegements
This repository is built upon the following repositories:   
- [When-Adversarial-Training-Meets-Vision-Transformers](https://github.com/mo666666/When-Adversarial-Training-Meets-Vision-Transformers)
- [LipsFormer](https://github.com/IDEA-Research/LipsFormer)
- [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
- [vits-robustness-torch](https://github.com/dedeswim/vits-robustness-torch)




## Cite this work
If you find our code is useful, please cite our paper!
```bibtex
@inproceedings{hu2024specformer,
  title={SpecFormer: Guarding Vision Transformer Robustness via Maximum Singular Value Penalization},
  author={Hu, Xixu and Zheng, Runkai and Wang, Jindong and Leung, Cheukhang and Wu, Qi and Xie, Xing},
  booktitle={European Conference on Computer Vision},
  year={2024},
  organization={Springer}
}
```