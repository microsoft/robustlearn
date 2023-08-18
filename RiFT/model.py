# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random

from models import *


def create_model(model_name, input_size, num_classes, device, patch_size=4, resume=None):

	if model_name == "ResNet34":
		model = ResNet34(input_size, num_classes)
	elif model_name == "ResNet18":
		model = ResNet18(input_size, num_classes)
	elif model_name == "ResNet50":
		model = ResNet50(input_size, num_classes)
	elif model_name == "ResNet152":
		model = ResNet152(input_size, num_classes)
	elif model_name == "DenseNet":
		model = DenseNet121(input_size, num_classes)
	elif model_name == "DenseNet50":
		model = DenseNet50(input_size, num_classes)
	elif model_name == "VGG19":
		model = VGG("VGG19", input_size, num_classes)
	elif model_name == "WideResNet34":
		model = WideResNet(image_size=input_size, depth=34, widen_factor=10, num_classes=num_classes)
	elif model_name == "WideResNet28":
		model = WideResNet(image_size=input_size, depth=28, widen_factor=10, num_classes=num_classes)
	elif model_name == "WideResNet22_2":
		model = WideResNet(image_size=input_size, depth=22, widen_factor=2, num_classes=num_classes)
	elif model_name == 'WideResNet34_5':
		model = WideResNet(image_size=input_size, depth=34, widen_factor=5, num_classes=num_classes)
	elif model_name == "FCN":
		model = FCN()
		# for name in net.state_dict().keys():
		# 	print(name)
		# exit()

	elif model_name == "ViT":
	# ViT for cifar10
		model = ViT(image_size=input_size, patch_size=patch_size, num_classes=num_classes, dim=512, depth=6, heads=8, mlp_dim=512, dropout=0.1, emb_dropout=0.1)
	# elif model == "ConvMixer":
	# 	net = ConvMixer(dim=256, depth=16, kernel_size=8, patch_size=1, n_classes=num_classes)
	
	# elif model == "ViT-timm":
	# 	import timm
	# 	net = timm.create_model("vit_small_patch16_224", pretrained=True)
	# 	net.head = nn.Linear(net.head.in_features, num_classes)


	# else: 
	# 	# print("using effiecientnet")
	# 	# net = models.efficientnet_b3(pretrained=True)
		
	# 	from efficientnet_pytorch import EfficientNet
	# 	# net = EfficientNet.from_pretrained('efficientnet-b3', num_classes=200)
	# 	net = EfficientNet.from_name('efficientnet-b3', num_classes=200)

	model = model.to(device)

	if device == 'cuda':
		model = torch.nn.DataParallel(model)

	if resume is not None:
		print(resume)
		checkpoint = torch.load(resume)
		# print(checkpoint.items())
		# exit()
		if "net" in checkpoint.keys():
			model.load_state_dict(checkpoint["net"])
		elif "state_dict" in checkpoint.keys():
			model.load_state_dict(checkpoint["state_dict"])
		elif "model" in checkpoint.keys():
			model.load_state_dict(checkpoint["model"])
		else:
			model.load_state_dict(checkpoint)

	return model


