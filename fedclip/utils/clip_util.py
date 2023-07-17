# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import clip
import torch


def get_similarity(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    return similarity


def get_image_features(image, model, cpreprocess, device='cuda', need_preprocess=False):
    if need_preprocess:
        image = cpreprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features


def freeze_param(model):
    for name, param in model.named_parameters():
        param.requires_grad = False


def get_text_features_list(texts, model, device='cuda', train=False):
    if train:
        text_inputs = torch.cat([clip.tokenize(c)
                                for c in texts]).to(device)
        text_features = model.encode_text(text_inputs)
    else:
        with torch.no_grad():
            text_inputs = torch.cat([clip.tokenize(c)
                                     for c in texts]).to(device)
            text_features = model.encode_text(text_inputs)

    return text_features
