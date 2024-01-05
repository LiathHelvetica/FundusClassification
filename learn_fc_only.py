from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

from constants import BATCH_SIZES, TRAIN_AUGMENT_PATH, TRAIN_LABELS_PATH, EXCLUDED_LABELS, VALIDATION_AUGMENT_PATH, \
  VALIDATION_LABELS_PATH, EPOCH_VALUES, CSV_HEADERS, TRAIN_DATA_OUT_FILE, LABEL_DICT_OUT_PATH
from dataset import FundusImageDataset
from itertools import product

from no_op_scheduler import NoOpScheduler
from utils import update_resizing


def model_last_layer_fc(f_model_create, device, classes, x, y):
  def op():
    model = f_model_create()
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)
    return model, x, y
  return op


def model_last_layer_sequential_classifier(f_model_create, device, classes, x, y):
  def op():
    model = f_model_create()
    fc = model.classifier[-1]
    model.classifier[-1] = nn.Linear(fc.in_features, len(classes))
    model.to(device)
    return model, x, y
  return op


def model_last_layer_sequential_heads(f_model_create, device, classes, x, y):
  def op():
    model = f_model_create()
    fc = model.heads[-1]
    model.heads[-1] = nn.Linear(fc.in_features, len(classes))
    model.to(device)
    return model, x, y
  return op


def model_last_layer_classifier(f_model_create, device, classes, x, y):
  def op():
    model = f_model_create()
    model.classifier = nn.Linear(model.classifier.in_features, len(classes))
    model.to(device)
    return model, x, y
  return op


def model_last_layer_head(f_model_create, device, classes, x, y):
  def op():
    model = f_model_create()
    model.head = nn.Linear(model.head.in_features, len(classes))
    model.to(device)
    return model, x, y
  return op


train_ds = FundusImageDataset(TRAIN_AUGMENT_PATH, TRAIN_LABELS_PATH, EXCLUDED_LABELS)
val_ds = FundusImageDataset(VALIDATION_AUGMENT_PATH, VALIDATION_LABELS_PATH, EXCLUDED_LABELS)
labels_l = list(set(train_ds.local_labels + val_ds.local_labels))
train_ds.set_labels(labels_l)
val_ds.set_labels(labels_l)
classes = torch.arange(len(labels_l))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_initializers = [
  model_last_layer_fc(lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1), device, classes, 342, 342),
  model_last_layer_fc(lambda: models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.regnet_y_1_6gf(weights=models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.regnet_y_1_6gf(weights=models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.regnet_y_3_2gf(weights=models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.regnet_y_3_2gf(weights=models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.regnet_y_8gf(weights=models.RegNet_Y_8GF_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.regnet_y_8gf(weights=models.RegNet_Y_8GF_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1), device, classes, 384, 384),
  model_last_layer_fc(lambda: models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_LINEAR_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_LINEAR_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1), device, classes, 384, 384),
  model_last_layer_fc(lambda: models.regnet_y_128gf(weights=models.RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1), device, classes, 384, 384),
  model_last_layer_fc(lambda: models.regnet_y_128gf(weights=models.RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_LINEAR_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.regnet_x_400mf(weights=models.RegNet_X_400MF_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.regnet_x_400mf(weights=models.RegNet_X_400MF_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.regnet_x_800mf(weights=models.RegNet_X_800MF_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.regnet_x_800mf(weights=models.RegNet_X_800MF_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.regnet_x_1_6gf(weights=models.RegNet_X_1_6GF_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.regnet_x_1_6gf(weights=models.RegNet_X_1_6GF_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.regnet_x_3_2gf(weights=models.RegNet_X_3_2GF_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.regnet_x_3_2gf(weights=models.RegNet_X_3_2GF_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.regnet_x_8gf(weights=models.RegNet_X_8GF_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.regnet_x_8gf(weights=models.RegNet_X_8GF_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.regnet_x_16gf(weights=models.RegNet_X_16GF_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.regnet_x_16gf(weights=models.RegNet_X_16GF_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.regnet_x_32gf(weights=models.RegNet_X_32GF_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.regnet_x_32gf(weights=models.RegNet_X_32GF_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.resnext101_64x4d(weights=models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.shufflenet_v2_x1_5(weights=models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  model_last_layer_fc(lambda: models.wide_resnet101_2(weights=models.Wide_ResNet101_2_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_fc(lambda: models.wide_resnet101_2(weights=models.Wide_ResNet101_2_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  ####
  model_last_layer_sequential_classifier(lambda: models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_sequential_classifier(lambda: models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1), device, classes, 236, 236),
  model_last_layer_sequential_classifier(lambda: models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1), device, classes, 230, 230),
  model_last_layer_sequential_classifier(lambda: models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1), device, classes, 232, 232),
  model_last_layer_sequential_classifier(lambda: models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1), device, classes, 232, 232),
  model_last_layer_sequential_classifier(lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1), device, classes, 256, 256),
  model_last_layer_sequential_classifier(lambda: models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1), device, classes, 256, 256),
  model_last_layer_sequential_classifier(lambda: models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1), device, classes, 288, 288),
  model_last_layer_sequential_classifier(lambda: models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1), device, classes, 320, 320),
  model_last_layer_sequential_classifier(lambda: models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1), device, classes, 384, 384),
  model_last_layer_sequential_classifier(lambda: models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1), device, classes, 456, 456),
  model_last_layer_sequential_classifier(lambda: models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.IMAGENET1K_V1), device, classes, 528, 528),
  model_last_layer_sequential_classifier(lambda: models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1), device, classes, 600, 600),
  model_last_layer_sequential_classifier(lambda: models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1), device, classes, 384, 384),
  model_last_layer_sequential_classifier(lambda: models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1), device, classes, 480, 480),
  model_last_layer_sequential_classifier(lambda: models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1), device, classes, 480, 480),
  model_last_layer_sequential_classifier(lambda: models.maxvit_t(weights=models.MaxVit_T_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_sequential_classifier(lambda: models.mnasnet0_5(weights=models.MNASNet0_5_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_sequential_classifier(lambda: models.mnasnet0_75(weights=models.MNASNet0_75_Weights.IMAGENET1K_V1), device, classes, 232, 232),
  model_last_layer_sequential_classifier(lambda: models.mnasnet1_0(weights=models.MNASNet1_0_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_sequential_classifier(lambda: models.mnasnet1_3(weights=models.MNASNet1_3_Weights.IMAGENET1K_V1), device, classes, 232, 232),
  model_last_layer_sequential_classifier(lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_sequential_classifier(lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  model_last_layer_sequential_classifier(lambda: models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_sequential_classifier(lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_sequential_classifier(lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2), device, classes, 232, 232),
  model_last_layer_sequential_classifier(lambda: models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_sequential_classifier(lambda: models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_sequential_classifier(lambda: models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_sequential_classifier(lambda: models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_sequential_classifier(lambda: models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_sequential_classifier(lambda: models.vgg13_bn(weights=models.VGG13_BN_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_sequential_classifier(lambda: models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_sequential_classifier(lambda: models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  ####
  model_last_layer_sequential_heads(lambda: models.vit_h_14(weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1), device, classes, 518, 518),
  model_last_layer_sequential_heads(lambda: models.vit_h_14(weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1), device, classes, 224, 224),
  model_last_layer_sequential_heads(lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1), device, classes, 224, 224),
  model_last_layer_sequential_heads(lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_sequential_heads(lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1), device, classes, 384, 384),
  model_last_layer_sequential_heads(lambda: models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1), device, classes, 242, 242),
  model_last_layer_sequential_heads(lambda: models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1), device, classes, 512, 512),
  model_last_layer_sequential_heads(lambda: models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1), device, classes, 224, 224),
  model_last_layer_sequential_heads(lambda: models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_sequential_heads(lambda: models.vit_l_32(weights=models.ViT_L_32_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  ####
  model_last_layer_classifier(lambda: models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_classifier(lambda: models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_classifier(lambda: models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  model_last_layer_classifier(lambda: models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1), device, classes, 224, 224),
  ####
  model_last_layer_head(lambda: models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1), device, classes, 238, 238),
  model_last_layer_head(lambda: models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1), device, classes, 232, 232),
  model_last_layer_head(lambda: models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1), device, classes, 246, 246),
  model_last_layer_head(lambda: models.swin_v2_b(weights=models.Swin_V2_B_Weights.IMAGENET1K_V1), device, classes, 272, 272),
  model_last_layer_head(lambda: models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1), device, classes, 260, 260),
  model_last_layer_head(lambda: models.swin_v2_s(weights=models.Swin_V2_S_Weights.IMAGENET1K_V1), device, classes, 260, 260)
]

loss_functions = [
  nn.CrossEntropyLoss(),
  nn.BCELoss(),
  nn.MarginRankingLoss(),
  # nn.MarginRankingLoss(margin=0.1),
  # nn.MarginRankingLoss(margin=0.5),
  # nn.MarginRankingLoss(margin=1.0),
  # nn.MarginRankingLoss(margin=2.0),
  # nn.MultiMarginLoss(p=1),
  # nn.MultiMarginLoss(p=2),
  # nn.SmoothL1Loss(),
  # nn.NLLLoss(),
  # nn.HingeEmbeddingLoss(margin=0.1),
  # nn.HingeEmbeddingLoss(margin=0.5),
  # nn.HingeEmbeddingLoss(margin=1.0),
  # nn.HingeEmbeddingLoss(margin=2.0),
  # nn.TripletMarginLoss()  # tweakable
]

# experiment with more optimisers
optimizers = [
  lambda params: optim.SGD(params, lr=0.1, momentum=0.9),
  lambda params: optim.Adam(params, lr=0.001),
  lambda params: optim.Adagrad(params, lr=0.01),
  # lambda params: optim.RMSprop(params, lr=0.01, momentum=0.1),
  # lambda params: optim.RMSprop(params, lr=0.01, momentum=0.1),
  # lambda params: optim.Adadelta(params)
]

schedulers = [
  lambda opt, n_epochs: NoOpScheduler(),
  lambda opt, n_epochs: lr_scheduler.StepLR(optimizer=opt, step_size=int(n_epochs / 2), gamma=0.1),
  lambda opt, n_epochs: lr_scheduler.StepLR(optimizer=opt, step_size=int(n_epochs / 3), gamma=0.1),
  lambda opt, n_epochs: lr_scheduler.StepLR(optimizer=opt, step_size=int(n_epochs / 4), gamma=0.1),
  lambda opt, n_epochs: lr_scheduler.StepLR(optimizer=opt, step_size=int(n_epochs / 4), gamma=0.5)
  # experiment with ReduceLROnPlateau - seems promising
]


def try_or_else(getter, default):
  try:
    return getter()
  except:
    return default


def get_model_data(best_acc, epochs, criterion, optimizer, model, scheduler, tdelta):
  return {
    CSV_HEADERS[0]: best_acc,
    CSV_HEADERS[1]: epochs,
    CSV_HEADERS[2]: type(criterion).__name__,
    CSV_HEADERS[3]: type(optimizer).__name__,
    CSV_HEADERS[4]: optimizer.lr,
    CSV_HEADERS[5]: try_or_else(lambda: optimizer.momentum, "no momentum for optimizer"),
    CSV_HEADERS[6]: type(model.weights).__name__,
    CSV_HEADERS[7]: type(scheduler).__name__,
    CSV_HEADERS[8]: try_or_else(scheduler.step_size, "no-op"),
    CSV_HEADERS[9]: try_or_else(scheduler.gamma, "no-op"),
    CSV_HEADERS[10]: str(tdelta)
  }


with open(TRAIN_DATA_OUT_FILE, "w") as f_out:

  f_out.write(",".join(CSV_HEADERS))

  for batch_size in BATCH_SIZES:

    dataloaders = {
      "train": DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
      ),
      "val": DataLoader(
        val_ds, batch_size=batch_size, shuffle=True
      )
    }

    for model_f, epochs, loss_f, optim_f, schedul_f in product(model_initializers, EPOCH_VALUES, loss_functions, optimizers, schedulers):
      # train loop
      start = datetime.now()
      model, x_size, y_size = model_f()
      update_resizing([train_ds, val_ds], x_size, y_size)
      criterion = loss_f
      optimizer = optim_f(model.parameters())
      scheduler = schedul_f(optimizer, epochs)
      best_acc = 0.0
      for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0
        model.train()
        for inputs, labels in dataloaders["train"]:
          inputs = inputs.to(device)
          labels = labels.to(device)
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          scheduler.step()
        model.eval()
        epoch_acc = -1.0
        for inputs, labels in dataloaders["val"]:
          inputs = inputs.to(device)
          labels = labels.to(device)
          optimizer.zero_grad()
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)
          running_loss += loss.item() * inputs.size(0)
          running_corrects += torch.sum(preds == labels.data)
          epoch_loss = running_loss / len(val_ds)
          epoch_acc = running_corrects / len(val_ds)
        if epoch_acc > best_acc:
          best_acc = epoch_acc
      stop = datetime.now()
      model_data = get_model_data(best_acc, epochs, criterion, optimizer, model, scheduler, stop - start)
      print(model_data)
      f_out.write(",".join(map(lambda header: model_data[header], CSV_HEADERS)))
