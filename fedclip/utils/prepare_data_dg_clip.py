# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import clip
import torchvision.datasets as datasets
from PIL import ImageFile
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import sys
import os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageTextData(object):

    def __init__(self, dataset, root, preprocess, prompt='a picture of a'):
        dataset = os.path.join(root, dataset)
        data = datasets.ImageFolder(dataset, transform=self._TRANSFORM)
        labels = data.classes
        self.data = data
        self.labels = labels
        if prompt:
            self.labels = [prompt + ' ' + x for x in self.labels]

        self.preprocess = preprocess
        self.text = clip.tokenize(self.labels)

    def __getitem__(self, index):
        image, label = self.data.imgs[index]
        if self.preprocess is not None:
            image = self.preprocess(Image.open(image))
        text_enc = self.text[label]
        return image, text_enc, label

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_data_name_by_index(index):
        name = ImageTextData._DATA_FOLDER[index]
        name = name.replace('/', '_')
        return name

    _TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_data(data_name):
    datalist = {'pacs': 'pacs'}
    if datalist[data_name] not in globals():
        raise NotImplementedError("Dataset not found: {}".format(data_name))
    return globals()[datalist[data_name]]


def getfeadataloader(args, model):
    trl, val, tel = [], [], []
    trd, vad, ted = [], [], []
    for i, item in enumerate(args.domains):
        if i in args.test_envs:
            data = ImageTextData(
                item, args.root_dir+args.dataset+'/', model.preprocess)
            model.setselflabel(data.labels)
            ted.append(torch.utils.data.DataLoader(
                data, batch_size=args.batch, shuffle=False))
            trd.append(0)
            vad.append(0)
        else:
            data = ImageTextData(
                item, args.root_dir+args.dataset+'/', model.preprocess)
            l = len(data)
            index = np.arange(l)
            np.random.seed(args.seed)
            np.random.shuffle(index)
            l1, l2, l3 = int(l*args.datapercent), int(l*0.2), int(l*0.2)
            trl.append(torch.utils.data.Subset(data, index[:l1]))
            val.append(torch.utils.data.Subset(data, index[l1:l1+l2]))
            tel.append(torch.utils.data.Subset(data, index[l1+l2:l1+l2+l3]))
            trd.append(torch.utils.data.DataLoader(
                trl[-1], batch_size=args.batch, shuffle=True))
            vad.append(torch.utils.data.DataLoader(
                val[-1], batch_size=args.batch, shuffle=False))
            ted.append(torch.utils.data.DataLoader(
                tel[-1], batch_size=args.batch, shuffle=False))
    return trd, vad, ted


def pacs(args, model):
    trd, vad, ted = getfeadataloader(args, model)
    return trd, vad, ted
