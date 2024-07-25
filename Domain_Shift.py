from utils.logger import setup_logger
from datasets.make_dataloader_stage3 import make_dataloader
from model.make_model_clipreid import make_model
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor_stage3 import do_train_stage3

import random
import torch
import numpy as np
import os
import argparse
from config import cfg

"""
stage 3:
1. support set
2. student image_encoder training
"""
"""
modify:
1.train.py -> Domain_Shift.py
2.processor_clipreid_stage2.py -> processor_stage3.py
3.make_loss.py -> make_loss_stage3.py
4.make_dataloader_clipreid.py -> make_dataloader_stage3.py
"""

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

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
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # 先读取要使用的超参数
    train_loader_stage2, train_loader_stage1, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # teacher
    model1 = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model1.load_param(cfg.TEST.WEIGHT)

    # stage3 仅读取的是Target Domain的数据
    # 但是 train_loader_stage2 中使用的是model中的序号 用于之后取text_features
    #
    # val测试阶段全部使用target domain的全部标签数据
    # 测试阶段的代码暂时都不需要修改
    train_loader_stage2, train_loader_stage1, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg, model1)

    # student
    model2 = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)






