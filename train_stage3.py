from utils.logger import setup_logger
from datasets.make_dataloader_stage3 import make_dataloader_stage3, get_parameter
from model.make_model_clipreid import make_model
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
from torch import nn
import torch.nn.functional as F

'''
modify:
仅用一个model
'''

'''
train_stage3.py 做迁移学习的训练
input: support_set (text_features, images)
        stage2_model的trained_image_encoder as teacher_model
        stage1_model的image_encoder as student_model
output: trained student_model
progress: 1.dataloader the aim support
            2.实例化teacher_model.eval() from stage2 & source domain
            3.实例化student_model from stage1 需要含有text_features
            4.support的图片分别注入两个model的image_encoder,获得两个text_features
            5.计算两个text_features的kl_loss
            6.利用kl_loss和lab_loss（一个text_features多张图片）去优化student_model teacher_model的参数不优化
            7.student_model.save
        7.target domain上测试student_model
'''

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/vit_clipreid.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # support set
    train_loader_stage3 = make_dataloader_stage3(cfg)

    # how to use dataloader
    # for inputs, labels in train_loader_stage3:
    #     # inputs(batch_size,3,256,128) # lable(batch_size,)
    #     inputs.to("cuda")
    #     labels.to("cuda")

    num_classes, cam_num, view_num = get_parameter(cfg)

    # # 2.实例化teacher_model.eval() from stage2 & source domain
    # teacher_model = make_model(cfg, num_class=num_classes, camera_num=cam_num, view_num =view_num)
    # teacher_model.load_param(cfg.TEST.WEIGHT)
    # teacher_model.eval()
    # teacher_model.to("cuda")

    # 3.实例化student_model from stage1 clip.image_encoder 用的一定是不包含target domain任何分类数据的model
    student_model = make_model(cfg, num_class=num_classes, camera_num=cam_num, view_num =view_num)
    student_model.load_param(cfg.DATASETS.WEIGHT)
    student_model.to("cuda")

    kl_loss = nn.KLDivLoss(reduction="batchmean")

    # 添加优化器 可修改
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-5)

    lower_loss = 1

    # 开始训练
    for epoch in range(cfg.Stage3.MAX_EPOCHS):
        for imgs, labels in train_loader_stage3:
            # inputs(batch_size,3,256,128) # lable(batch_size,)
            imgs = imgs.to("cuda")

            # 两个image_features都已经归一化
            with torch.no_grad():
                teacher_image_features = F.softmax(teacher_model(x=imgs, get_image=True))  # (batch,512)
            student_image_features = F.softmax(student_model(x=imgs, get_image=True))  # (batch,512)

            # 计算loss
            loss = kl_loss(student_image_features.log(), teacher_image_features)

            if loss < lower_loss:
                lower_loss = loss
                torch.save(student_model.state_dict(), cfg.OUTPUT_DIR + "/" + "stage3_best_" + cfg.DATASETS.NAMES + "_2_" + cfg.Domain_Shift_DATASETS.NAMES + "_"+'_student_model.pth')

                # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 每个epoch结束后，打印loss值
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # 保存模型
    torch.save(student_model.state_dict(), cfg.OUTPUT_DIR + "/" + "stage3_best_" + cfg.DATASETS.NAMES+"_2_"+cfg.Domain_Shift_DATASETS.NAMES+"_"+str(epoch)+'_student_model.pth')







    print("1")







