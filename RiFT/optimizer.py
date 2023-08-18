# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.optim as optim


def create_optimizer(optim_name, net, lr, momentum, weight_decay=0):
	if optim_name == "Adam":
		optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
	elif optim_name == "RMSprop":
		optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=weight_decay) 
	elif optim_name == "AdamW":
		optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
	else:
		optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

	return optimizer


def create_scheduler(args, optimizer, lr_decays=None):

	if args.lr_scheduler == "step":
		if lr_decays is None:
			lr_decays = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decays, gamma=args.lr_decay_gamma, last_epoch=-1)
	elif args.lr_scheduler == "cosine":
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
	else:
		raise ValueError("The scheduler is not implemented!")
	# elif args.lr_scheduler == "cyclic":
	# 	pass
	return scheduler