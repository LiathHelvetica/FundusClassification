import gc
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models

from constants import BATCH_SIZES, TRAIN_LABELS_PATH, EXCLUDED_LABELS, \
  VALIDATION_LABELS_PATH, CSV_HEADERS, TRAIN_DATA_OUT_FILE, TRAIN_224_AUGMENT_PATH, \
  VALIDATION_224_AUGMENT_PATH, EPOCHS
from correct_counter import CounterCollection
from dataset import FundusImageDataset
from itertools import product

from no_op_scheduler import NoOpScheduler
from utils import update_resizing


def model_last_layer_fc(f_model_create, device, classes, x, y, m_name):
  def op():
    model = f_model_create()
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)
    return model, x, y, m_name
  return op


def model_last_layer_sequential_classifier(f_model_create, device, classes, x, y, m_name):
  def op():
    model = f_model_create()
    fc = model.classifier[-1]
    model.classifier[-1] = nn.Linear(fc.in_features, len(classes))
    model.to(device)
    return model, x, y, m_name
  return op


def model_last_layer_sequential_heads(f_model_create, device, classes, x, y, m_name):
  def op():
    model = f_model_create()
    fc = model.heads[-1]
    model.heads[-1] = nn.Linear(fc.in_features, len(classes))
    model.to(device)
    return model, x, y, m_name
  return op


def model_last_layer_classifier(f_model_create, device, classes, x, y, m_name):
  def op():
    model = f_model_create()
    model.classifier = nn.Linear(model.classifier.in_features, len(classes))
    model.to(device)
    return model, x, y, m_name
  return op


def model_last_layer_head(f_model_create, device, classes, x, y, m_name):
  def op():
    model = f_model_create()
    model.head = nn.Linear(model.head.in_features, len(classes))
    model.to(device)
    return model, x, y, m_name
  return op


train_ds = FundusImageDataset(
  TRAIN_224_AUGMENT_PATH,
  TRAIN_LABELS_PATH,
  EXCLUDED_LABELS,
  max_per_class=2700,
  dont_resize=True,
  do_shuffle=True
)
val_ds = FundusImageDataset(
  VALIDATION_224_AUGMENT_PATH,
  VALIDATION_LABELS_PATH,
  EXCLUDED_LABELS,
  max_per_class=700,
  dont_resize=True,
  do_shuffle=True
)
labels_l = list(set(train_ds.local_labels + val_ds.local_labels))
train_ds.set_labels(labels_l)
val_ds.set_labels(labels_l)
classes = torch.arange(len(labels_l))
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = "cuda"

model_initializers = [
  model_last_layer_fc(lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1), device, classes, 224, 224,
                      "resnet50"),
  # model_last_layer_fc(lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2), device, classes, 232, 232, "resnet50"),
  model_last_layer_fc(lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1), device, classes, 224, 224,
                      "resnet18"),
  model_last_layer_fc(lambda: models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1), device, classes, 224, 224,
                      "resnet34"),
  model_last_layer_fc(lambda: models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1), device, classes, 224,
                      224, "resnet101"),
  # model_last_layer_fc(lambda: models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2), device, classes, 232, 232, "resnet101"),
  model_last_layer_fc(lambda: models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1), device, classes, 224,
                      224, "resnet152"),
  # model_last_layer_fc(lambda: models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2), device, classes, 232, 232, "resnet152"),
  model_last_layer_fc(lambda: models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1), device, classes, 224,
                      224, "googlenet"),
  # model_last_layer_fc(lambda: models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1), device, classes, 342, 342, "inception_v3"),
  model_last_layer_fc(lambda: models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V1), device,
                      classes, 224, 224, "regnet_y_400mf"),
  # model_last_layer_fc(lambda: models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V2), device, classes, 232, 232, "regnet_y_400mf"),
  model_last_layer_fc(lambda: models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.IMAGENET1K_V1), device,
                      classes, 224, 224, "regnet_y_800mf"),
  # model_last_layer_fc(lambda: models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.IMAGENET1K_V2), device, classes, 232, 232, "regnet_y_800mf"),
  model_last_layer_fc(lambda: models.regnet_y_1_6gf(weights=models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V1), device,
                      classes, 224, 224, "regnet_y_1_6gf"),
  # model_last_layer_fc(lambda: models.regnet_y_1_6gf(weights=models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V2), device, classes, 232, 232, "regnet_y_1_6gf"),
  model_last_layer_fc(lambda: models.regnet_y_3_2gf(weights=models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V1), device,
                      classes, 224, 224, "regnet_y_3_2gf"),
  # model_last_layer_fc(lambda: models.regnet_y_3_2gf(weights=models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V2), device, classes, 232, 232, "regnet_y_3_2gf"),
  model_last_layer_fc(lambda: models.regnet_y_8gf(weights=models.RegNet_Y_8GF_Weights.IMAGENET1K_V1), device, classes,
                      224, 224, "regnet_y_8gf"),
  # model_last_layer_fc(lambda: models.regnet_y_8gf(weights=models.RegNet_Y_8GF_Weights.IMAGENET1K_V2), device, classes, 232, 232, "regnet_y_8gf"),
  model_last_layer_fc(lambda: models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_V1), device, classes,
                      224, 224, "regnet_y_16gf"),
  # model_last_layer_fc(lambda: models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_V2), device, classes, 232, 232, "regnet_y_16gf"),
  # model_last_layer_fc(lambda: models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1), device, classes, 384, 384, "regnet_y_16gf"),
  model_last_layer_fc(lambda: models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_LINEAR_V1),
                      device, classes, 224, 224, "regnet_y_16gf"),
  model_last_layer_fc(lambda: models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_V1), device, classes,
                      224, 224, "regnet_y_32gf"),
  # model_last_layer_fc(lambda: models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_V2), device, classes, 232, 232, "regnet_y_32gf"),
  model_last_layer_fc(lambda: models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_LINEAR_V1),
                      device, classes, 224, 224, "regnet_y_32gf"),
  # model_last_layer_fc(lambda: models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1), device, classes, 384, 384, "regnet_y_32gf"),
  # model_last_layer_fc(lambda: models.regnet_y_128gf(weights=models.RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1), device, classes, 384, 384, "regnet_y_128gf"),
  model_last_layer_fc(lambda: models.regnet_y_128gf(weights=models.RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_LINEAR_V1),
                      device, classes, 224, 224, "regnet_y_128gf"),
  model_last_layer_fc(lambda: models.regnet_x_400mf(weights=models.RegNet_X_400MF_Weights.IMAGENET1K_V1), device,
                      classes, 224, 224, "regnet_x_400mf"),
  # model_last_layer_fc(lambda: models.regnet_x_400mf(weights=models.RegNet_X_400MF_Weights.IMAGENET1K_V2), device, classes, 232, 232, "regnet_x_400mf"),
  model_last_layer_fc(lambda: models.regnet_x_800mf(weights=models.RegNet_X_800MF_Weights.IMAGENET1K_V1), device,
                      classes, 224, 224, "regnet_x_800mf"),
  # model_last_layer_fc(lambda: models.regnet_x_800mf(weights=models.RegNet_X_800MF_Weights.IMAGENET1K_V2), device, classes, 232, 232, "regnet_x_800mf"),
  model_last_layer_fc(lambda: models.regnet_x_1_6gf(weights=models.RegNet_X_1_6GF_Weights.IMAGENET1K_V1), device,
                      classes, 224, 224, "regnet_x_1_6gf"),
  # model_last_layer_fc(lambda: models.regnet_x_1_6gf(weights=models.RegNet_X_1_6GF_Weights.IMAGENET1K_V2), device, classes, 232, 232, "regnet_x_1_6gf"),
  model_last_layer_fc(lambda: models.regnet_x_3_2gf(weights=models.RegNet_X_3_2GF_Weights.IMAGENET1K_V1), device,
                      classes, 224, 224, "regnet_x_3_2gf"),
  # model_last_layer_fc(lambda: models.regnet_x_3_2gf(weights=models.RegNet_X_3_2GF_Weights.IMAGENET1K_V2), device, classes, 232, 232, "regnet_x_3_2gf"),
  model_last_layer_fc(lambda: models.regnet_x_8gf(weights=models.RegNet_X_8GF_Weights.IMAGENET1K_V1), device, classes,
                      224, 224, "regnet_x_8gf"),
  # model_last_layer_fc(lambda: models.regnet_x_8gf(weights=models.RegNet_X_8GF_Weights.IMAGENET1K_V2), device, classes, 232, 232, "regnet_x_8gf"),
  model_last_layer_fc(lambda: models.regnet_x_16gf(weights=models.RegNet_X_16GF_Weights.IMAGENET1K_V1), device, classes,
                      224, 224, "regnet_x_16gf"),
  # model_last_layer_fc(lambda: models.regnet_x_16gf(weights=models.RegNet_X_16GF_Weights.IMAGENET1K_V2), device, classes, 232, 232, "regnet_x_16gf"),
  model_last_layer_fc(lambda: models.regnet_x_32gf(weights=models.RegNet_X_32GF_Weights.IMAGENET1K_V1), device, classes,
                      224, 224, "regnet_x_32gf"),
  # model_last_layer_fc(lambda: models.regnet_x_32gf(weights=models.RegNet_X_32GF_Weights.IMAGENET1K_V2), device, classes, 232, 232, "regnet_x_32gf"),
  model_last_layer_fc(lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1), device, classes, 224, 224,
                      "resnet18"),
  model_last_layer_fc(lambda: models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1), device, classes, 224, 224,
                      "resnet34"),
  model_last_layer_fc(lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1), device, classes, 224, 224,
                      "resnet50"),
  # model_last_layer_fc(lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2), device, classes, 232, 232, "resnet50"),
  model_last_layer_fc(lambda: models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1), device, classes, 224,
                      224, "resnet101"),
  model_last_layer_fc(lambda: models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2), device, classes, 232,
                      232, "resnet101"),
  model_last_layer_fc(lambda: models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1), device, classes, 224,
                      224, "resnet152"),
  # model_last_layer_fc(lambda: models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2), device, classes, 232, 232, "resnet152"),
  model_last_layer_fc(lambda: models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1), device,
                      classes, 224, 224, "resnext50_32x4d"),
  # model_last_layer_fc(lambda: models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2), device, classes, 232, 232, "resnext50_32x4d"),
  model_last_layer_fc(lambda: models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1), device,
                      classes, 224, 224, "resnext101_32x8d"),
  # model_last_layer_fc(lambda: models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2), device, classes, 232, 232, "resnext101_32x8d"),
  # model_last_layer_fc(lambda: models.resnext101_64x4d(weights=models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1), device, classes, 232, 232, "resnext101_64x4d"),
  model_last_layer_fc(lambda: models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1),
                      device, classes, 224, 224, "shufflenet_v2_x0_5"),
  model_last_layer_fc(lambda: models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1),
                      device, classes, 224, 224, "shufflenet_v2_x1_0"),
  # model_last_layer_fc(lambda: models.shufflenet_v2_x1_5(weights=models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1), device, classes, 232, 232, "shufflenet_v2_x1_5"),
  # model_last_layer_fc(lambda: models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1), device, classes, 232, 232, "shufflenet_v2_x2_0"),
  model_last_layer_fc(lambda: models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1), device,
                      classes, 224, 224, "wide_resnet50_2"),
  # model_last_layer_fc(lambda: models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2), device, classes, 232, 232, "wide_resnet50_2"),
  model_last_layer_fc(lambda: models.wide_resnet101_2(weights=models.Wide_ResNet101_2_Weights.IMAGENET1K_V1), device,
                      classes, 224, 224, "wide_resnet101_2"),
  # model_last_layer_fc(lambda: models.wide_resnet101_2(weights=models.Wide_ResNet101_2_Weights.IMAGENET1K_V2), device, classes, 232, 232, "wide_resnet101_2"),
  ####
  model_last_layer_sequential_classifier(lambda: models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1), device,
                                         classes, 224, 224, "alexnet"),
  # model_last_layer_sequential_classifier(lambda: models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1), device, classes, 236, 236, "convnext_tiny"),
  # model_last_layer_sequential_classifier(lambda: models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1), device, classes, 230, 230, "convnext_small"),
  # model_last_layer_sequential_classifier(lambda: models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1), device, classes, 232, 232, "convnext_base"),
  # model_last_layer_sequential_classifier(lambda: models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1), device, classes, 232, 232, "convnext_large"),
  # model_last_layer_sequential_classifier(lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1), device, classes, 256, 256, "efficientnet_b0"),
  # model_last_layer_sequential_classifier(lambda: models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1), device, classes, 256, 256, "efficientnet_b1"),
  # model_last_layer_sequential_classifier(lambda: models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1), device, classes, 288, 288, "efficientnet_b2"),
  # model_last_layer_sequential_classifier(lambda: models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1), device, classes, 320, 320, "efficientnet_b3"),
  # model_last_layer_sequential_classifier(lambda: models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1), device, classes, 384, 384, "efficientnet_b4"),
  # model_last_layer_sequential_classifier(lambda: models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1), device, classes, 456, 456, "efficientnet_b5"),
  # model_last_layer_sequential_classifier(lambda: models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.IMAGENET1K_V1), device, classes, 528, 528, "efficientnet_b6"),
  # model_last_layer_sequential_classifier(lambda: models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1), device, classes, 600, 600, "efficientnet_b7"),
  # model_last_layer_sequential_classifier(lambda: models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1), device, classes, 384, 384, "efficientnet_v2_s"),
  # model_last_layer_sequential_classifier(lambda: models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1), device, classes, 480, 480, "efficientnet_v2_m"),
  # model_last_layer_sequential_classifier(lambda: models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1), device, classes, 480, 480, "efficientnet_v2_l"),
  model_last_layer_sequential_classifier(lambda: models.maxvit_t(weights=models.MaxVit_T_Weights.IMAGENET1K_V1), device,
                                         classes, 224, 224, "maxvit_t"),
  model_last_layer_sequential_classifier(lambda: models.mnasnet0_5(weights=models.MNASNet0_5_Weights.IMAGENET1K_V1),
                                         device, classes, 224, 224, "mnasnet0_5"),
  # model_last_layer_sequential_classifier(lambda: models.mnasnet0_75(weights=models.MNASNet0_75_Weights.IMAGENET1K_V1), device, classes, 232, 232, "mnasnet0_75"),
  model_last_layer_sequential_classifier(lambda: models.mnasnet1_0(weights=models.MNASNet1_0_Weights.IMAGENET1K_V1),
                                         device, classes, 224, 224, "mnasnet1_0"),
  # model_last_layer_sequential_classifier(lambda: models.mnasnet1_3(weights=models.MNASNet1_3_Weights.IMAGENET1K_V1), device, classes, 232, 232, "mnasnet1_3"),
  model_last_layer_sequential_classifier(lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1), device, classes, 224, 224, "mobilenet_v2"),
  # model_last_layer_sequential_classifier(lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2), device, classes, 232, 232, "mobilenet_v2"),
  model_last_layer_sequential_classifier(lambda: models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1), device, classes, 224, 224, "mobilenet_v3_small"),
  model_last_layer_sequential_classifier(lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1), device, classes, 224, 224, "mobilenet_v3_large"),
  # model_last_layer_sequential_classifier(lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2), device, classes, 232, 232, "mobilenet_v3_large"),
  model_last_layer_sequential_classifier(lambda: models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1), device, classes, 224, 224, "vgg11"),
  model_last_layer_sequential_classifier(lambda: models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1), device, classes, 224, 224, "vgg13"),
  model_last_layer_sequential_classifier(lambda: models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1), device, classes, 224, 224, "vgg16"),
  model_last_layer_sequential_classifier(lambda: models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1), device, classes, 224, 224, "vgg19"),
  model_last_layer_sequential_classifier(lambda: models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1), device, classes, 224, 224, "vgg11_bn"),
  model_last_layer_sequential_classifier(lambda: models.vgg13_bn(weights=models.VGG13_BN_Weights.IMAGENET1K_V1), device, classes, 224, 224, "vgg13_bn"),
  model_last_layer_sequential_classifier(lambda: models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1), device, classes, 224, 224, "vgg16_bn"),
  model_last_layer_sequential_classifier(lambda: models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1), device, classes, 224, 224, "vgg19_bn"),
  ####
  # model_last_layer_sequential_heads(lambda: models.vit_h_14(weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1), device, classes, 518, 518, "vit_h_14"),
  model_last_layer_sequential_heads(lambda: models.vit_h_14(weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1), device, classes, 224, 224, "vit_h_14"),
  model_last_layer_sequential_heads(lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1), device, classes, 224, 224, "vit_b_16"),
  model_last_layer_sequential_heads(lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1), device, classes, 224, 224, "vit_b_16"),
  # model_last_layer_sequential_heads(lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1), device, classes, 384, 384, "vit_b_16"),
  model_last_layer_sequential_heads(lambda: models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1), device, classes, 242, 242, "vit_l_16"),
  # model_last_layer_sequential_heads(lambda: models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1), device, classes, 512, 512, "vit_l_16"),
  model_last_layer_sequential_heads(lambda: models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1), device, classes, 224, 224, "vit_l_16"),
  model_last_layer_sequential_heads(lambda: models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1), device, classes, 224, 224, "vit_b_32"),
  model_last_layer_sequential_heads(lambda: models.vit_l_32(weights=models.ViT_L_32_Weights.IMAGENET1K_V1), device, classes, 224, 224, "vit_l_32"),
  ####
  model_last_layer_classifier(lambda: models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1), device, classes, 224, 224, "densenet121"),
  model_last_layer_classifier(lambda: models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1), device, classes, 224, 224, "densenet161"),
  model_last_layer_classifier(lambda: models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1), device, classes, 224, 224, "densenet169"),
  model_last_layer_classifier(lambda: models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1), device, classes, 224, 224, "densenet201"),
  ####
  # model_last_layer_head(lambda: models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1), device, classes, 238, 238, "swin_b"),
  # model_last_layer_head(lambda: models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1), device, classes, 232, 232, "swin_t"),
  # model_last_layer_head(lambda: models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1), device, classes, 246, 246, "swin_s"),
  # model_last_layer_head(lambda: models.swin_v2_b(weights=models.Swin_V2_B_Weights.IMAGENET1K_V1), device, classes, 272, 272, "swin_v2_b"),
  # model_last_layer_head(lambda: models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1), device, classes, 260, 260, "swin_v2_t"),
  # model_last_layer_head(lambda: models.swin_v2_s(weights=models.Swin_V2_S_Weights.IMAGENET1K_V1), device, classes, 260, 260, "swin_v2_s"0)
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
  lambda params: optim.SGD(params, lr=0.001, momentum=0.9),
  lambda params: optim.SGD(params, lr=0.01, momentum=0.9),
  lambda params: optim.Adam(params, lr=0.001),
  lambda params: optim.Adam(params, lr=0.01),
  lambda params: optim.Adagrad(params, lr=0.01),
  lambda params: optim.Adagrad(params, lr=0.1),
  lambda params: optim.RMSprop(params, lr=0.01),
  lambda params: optim.RMSprop(params, lr=0.1),
  lambda params: optim.RMSprop(params, lr=0.01, momentum=0.1),
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


def get_model_data(acc, epochs, criterion, optimizer, m_name, scheduler, tdelta, loss, val_size, corrects_total, counters):
  return {
    CSV_HEADERS[0]: acc.item(),
    CSV_HEADERS[1]: epochs,
    CSV_HEADERS[2]: type(criterion).__name__,
    CSV_HEADERS[3]: type(optimizer).__name__,
    CSV_HEADERS[4]: optimizer.defaults["lr"],
    CSV_HEADERS[5]: try_or_else(lambda: optimizer.defaults["momentum"], "no momentum for optimizer"),
    CSV_HEADERS[6]: m_name,
    CSV_HEADERS[7]: type(scheduler).__name__,
    CSV_HEADERS[8]: try_or_else(lambda: scheduler.step_size, "no-op"),
    CSV_HEADERS[9]: try_or_else(lambda: scheduler.gamma, "no-op"),
    CSV_HEADERS[10]: str(tdelta),
    CSV_HEADERS[11]: loss,
    CSV_HEADERS[12]: val_size,
    CSV_HEADERS[13]: corrects_total,
    CSV_HEADERS[14]: f'"{str(counters)}"'
  }


if __name__ == "__main__":
  with open(TRAIN_DATA_OUT_FILE, "w") as f_out:

    f_out.write(",".join(CSV_HEADERS))
    f_out.write("\n")

    for batch_size in BATCH_SIZES:

      dataloaders = {
        "train": DataLoader(
          train_ds, batch_size=batch_size, shuffle=True  # shuffling done on train_ds as well
        ),
        "val": DataLoader(
          val_ds, batch_size=batch_size, shuffle=True
        )
      }

      val_healthy_label = val_ds.get_ok_label_id()

      for model_f, loss_f, optim_f, schedul_f in product(
        model_initializers, loss_functions,
        optimizers, schedulers
      ):
        # train loop
        start = datetime.now()
        model, x_size, y_size, m_name = model_f()
        update_resizing([train_ds, val_ds], x_size, y_size)
        criterion = loss_f
        optimizer = optim_f(model.parameters())
        scheduler = schedul_f(optimizer, EPOCHS)
        best_acc = 0.0
        best_loss = np.inf
        for epoch in range(EPOCHS):
          running_loss = 0.0
          running_corrects = 0
          running_counters = CounterCollection()
          model.train()
          n_batches = len(dataloaders['train'])
          i_batch = 1
          print(f"Train - {n_batches} batches")
          for inputs, labels in dataloaders["train"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i_batch % 10 == 0:
              print(f"{i_batch} / {n_batches}")
              torch.cuda.empty_cache()
              gc.collect()
            i_batch = i_batch + 1
          model.eval()
          epoch_acc = -1.0
          n_batches = len(dataloaders['val'])
          i_batch = 1
          print(f"Validate - {n_batches} batches")
          for inputs, labels in dataloaders["val"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # optimizer.zero_grad()
            outputs = model(inputs)
            # ↓ tensor of indices of class e.g. ([0, 4, 1]) is class 0, 4 and 1 for samples 1, 2, 3
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)  # yeah this is correct
            running_counters.update(labels.data, preds)
            epoch_loss = running_loss / len(val_ds)
            epoch_acc = running_corrects / len(val_ds)
            if i_batch % 10 == 0:
              print(f"{i_batch} / {n_batches}")
              torch.cuda.empty_cache()
              gc.collect()
            i_batch = i_batch + 1
          stop = datetime.now()
          model_data = get_model_data(
            epoch_acc,
            epoch,
            criterion,
            optimizer,
            m_name,
            scheduler,
            stop - start,
            epoch_loss,
            len(val_ds),
            running_corrects,
            running_counters
          )
          print(model_data)
          f_out.write(",".join(map(lambda header: str(model_data[header]), CSV_HEADERS)))
          f_out.write("\n")
          f_out.flush()
# {'acc': tensor(0.1000), 'epochs': 1, 'criterion': 'CrossEntropyLoss', 'optimizer': 'SGD', 'lr': 0.1, 'optimizer-momentum': 0.9, 'weights': 'resnet50', 'scheduler': 'NoOpScheduler', 'scheduler-step-size': 'no-op', 'scheduler-gamma': 'no-op', 'duration': '0:04:27.518109', 'loss': 243181.21354166666}
