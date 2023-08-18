![](https://files.mdnice.com/user/45288/023bf2cb-1685-43ce-bba8-1ba9b66f80b4.png)

# Improving Generalization of Adversarial Training via Robust Critical Fine-Tuning

This is the official implementation of ICCV2023 [Improving Generalization of Adversarial Training via Robust Critical Fine-Tuning](https://arxiv.org/abs/2308.02533).

**Abstract**: Deep neural networks are susceptible to adversarial examples, posing a significant security risk in critical applications. Adversarial Training (AT) is a well-established technique to enhance adversarial robustness, but it often comes at the cost of decreased generalization ability. This paper proposes Robustness Critical Fine-Tuning (RiFT), a novel approach to enhance generalization without compromising adversarial robustness. The core idea of RiFT is to exploit the redundant capacity for robustness by fine-tuning the adversarially trained model on its non-robust-critical module. To do so, we introduce module robust criticality (MRC), a measure that evaluates the significance of a given module to model robustness under worst-case weight perturbations. Using this measure, we identify the module with the lowest MRC value as the non-robust-critical module and fine-tune its weights to obtain fine-tuned weights. Subsequently, we linearly interpolate between the adversarially trained weights and fine-tuned weights to derive the optimal fine-tuned model weights. We demonstrate the efficacy of RiFT on ResNet18, ResNet34, and WideResNet34-10 models trained on CIFAR10, CIFAR100, and Tiny-ImageNet datasets. Our experiments show that RiFT can significantly improve both generalization and out-of-distribution robust- ness by around 1.5% while maintaining or even slightly enhancing adversarial robustness. Code is available at https://github.com/microsoft/robustlearn.

## Requirements

### Running Enviroments

To install requirements:

```
conda env create -f env.yaml
conda activate rift
```

### Datasets

CIFAR10 and CIFAR100 can be downloaded via PyTorch.

For other datasets:

1. [Tiny-ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
2. [CIFAR10-C](https://drive.google.com/drive/folders/1HDVw6CmX3HiG0ODFtI75iIfBDxSiSz2K)
3. [CIFAR100-C](https://drive.google.com/drive/folders/1HDVw6CmX3HiG0ODFtI75iIfBDxSiSz2K)
4. [Tiny-ImageNet-C](https://berkeley.app.box.com/s/6zt1qzwm34hgdzcvi45svsb10zspop8a)

After downloading these datasets, move them to ./data. The images in Tiny-ImageNet is 64x64 with 200 classes.

## Robust Critical Fine-Tuning

### Demo

Here we present a example for RiFT ResNet18 on CIFAR10.

Download the adversarially trained model weights [here](https://drive.google.com/drive/folders/1Uzqm1cOYFXLa97GZjjwfiVS2OcbpJK4o?usp=drive_link).

```
python main.py --layer=layer2.1.conv2 --resume="./ResNet18_CIFAR10.pth"
```

- layer: the desired layer name to fine-tune.

Here, layer2.1.conv2 is a non-robust-critical module.

The non-robust-critical module of each model on each dataset are summarized as follows:

|          | CIFAR10              | CIFAR100             | Tiny-ImageNet        |
| -------- | -------------------- | -------------------- | -------------------- |
| ResNet18 | layer2.1.conv2       | layer2.1.conv2       | layer3.1.conv2       |
| ResNet34 | layer2.3.conv2       | layer2.3.conv2       | layer3.5.conv2       |
| WRN34-10 | block1.layer.3.conv2 | block1.layer.2.conv2 | block1.layer.2.conv2 |

### Pipeline

1. Characterize the MRC for each module
   `python main.py --cal_mrc --resume=/path/to/your/model`
   This will output  the MRC for each module.
2. Fine-tuning on non-robust-critical module
   Based on the MRC output, choose a module with lowest MRC value to fine-tune. 
   We suggest to choose the **middle layers** according to our experience.
   Try different learning rate! Usually a small learning rate is preferred.
   `python main.py --layer=xxx --lr=yyy  --resume=zzz`
   When fine-tuning finish, it will automatically interpolate between adversarially trained weights and fine-tuned weights.
   The robust accuracy, in-distribution test acc are evaluated during the interpolation procedure.
3. Test OOD performance. Pick he best interpolation factor (the one with max IID generalization increase while not drop robustness so much.)
   `python eval_ood.py --resume=xxx`

## Results

![](https://files.mdnice.com/user/45288/c3c98491-a292-4888-82cc-081bc8d3c3c6.png)




![](https://files.mdnice.com/user/45288/bad5bb9f-788d-4350-ac5c-ddd850ade04f.png)



## References & Opensources

- Classification models [code](https://github.com/kuangliu/pytorch-cifar)
- Adversarial training [code](https://github.com/P2333/Bag-of-Tricks-for-AT)

## Contact

- Kaijie Zhu: kaijiezhu11@gmail.com
- Jindong Wang: jindongwang@outlook.com

## Citation

```
@inproceedings{zhu2023improving,
      title={Improving Generalization of Adversarial Training via Robust Critical Fine-Tuning}, 
     author={Zhu, Kaijie and Hu, Xixu and Wang, Jindong and Xie, Xing and Yang, Ge },
     year={2023},
	 booktitle={International Conference on Computer Vision},
}
```

