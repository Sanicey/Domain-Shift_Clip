# -*- coding: utf-8 -*-
import torch
import clip
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, ToTensor
import os
from config import cfg
import argparse
from datasets.make_dataloader_supportset import make_dataloader_supportset1, make_dataloader_supportset2
from model.make_model_clipreid import make_model
from processor.processor_clipreid_stage2 import do_inference
from utils.logger import setup_logger
import torch.nn as nn

'''
1.load the model of stage1 or stage2(just for the tokens in model.dict()), get the num_class and tokens(tokens better than text maybe)
2.build the folder, each source_domain_label have one folder
./support_set
-support_set
--source domain name
---0
----image1
----image2
---1
----image1
----image2
---2
......

3.only dataloader the images in target domain(just train images)
4.zero-shot, and put the aimed target_domain_train_img to aimed label folder -> support set
'''
'''
dataloader:
1.source domain's num_classes, camera_num, view_num
2.target domain's all images
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/vit_clipreid.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    num_classes, camera_num, view_num = make_dataloader_supportset1(cfg) # from source domain

    # # 爆显存
    # # 爆显存 分4个 这里用了10G显存？？
    num_of_splits = 100
    target_domain_all_images, image_set = make_dataloader_supportset2(cfg, num_of_splits) # from target domain

    # load model of stage1 or stage2(just for the tokens in model.dict()), get tokens
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_param(cfg.DATASETS.WEIGHT)
    model.to("cuda")

    image_features_list = []

    for imgs in target_domain_all_images:
        imgs = imgs.to("cuda")
        with torch.no_grad():
            score, feat, image_features = model(x=imgs)

        image_features_list.append(image_features.to("cpu"))

    image_features_list = torch.cat(image_features_list, dim=0) # tensor:(16522,512)

    # model_clip, _ = clip.load('ViT-B/16', "cuda") 只能使用(224,224)的图片输入

    # target_domain_all_images = make_dataloader_supportset2(cfg)
    # target_domain_all_images = target_domain_all_images.to("cuda") # 所有的target domain图片输入

    # score, feat, image_features : [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj
    # score, feat, image_features = model.image_encoder(target_domain_all_images)

    # 遍历 对每一个text_features找相似的图片
    for i in range(3):
        text_feature = model(label = torch.tensor([i]).to("cuda"), get_text = True).to("cpu") # (1,512) # 好像没做归一化？？

        # similarity = (3 * text_feature @ image_features_list.T).softmax(dim=-1) # (1,512)@(16522,512).T
        similarity = (3 * text_feature @ image_features_list.T)
        # # 阈值
        # indices = torch.where(similarity[0] > 0.05)[0]
        # values = similarity[0][indices]

        # 前5
        values, indices = torch.topk(similarity[0], 10)

        print("\nTop predictions:\n")
        for value, index in zip(values, indices):
            id = image_set[index]
            print(f"Index: {index}  匹配度: {1.0 * value.item():.2f}") # 这个不是百分比！！没有做softmax归一化！


    print("1")
    #




