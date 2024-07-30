from utils.logger import setup_logger
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor_clipreid_stage1 import do_train_stage1
from processor.processor_clipreid_stage2_supportset import do_train_stage2
from datasets.make_dataloader_stage3 import make_dataloader_stage3, make_stage3_testloader, get_parameter
import random
import torch
import numpy as np
import os
import argparse
from config import cfg

'''
train stage2 on support_set
    (1) dataloader: 1.support_set :imgs & labels
                    2.source domain: parameters for model_build, then load_parm from stage1 of source domain
                    3.target domain: val_loader for mAP & rk1
    (2) train image_encoder with imgs & text_features(get from model of stage1 by labels)
    (3) eval with trained_model on val_loader
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

    parser = argparse.ArgumentParser(description="Stage2 on support_set Training")
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

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # msmt17 -> market1501

    # 使用suorce domain的text_features
    #    support set的imgs
    #    target domain的val_loader
    #    用support set训练完之后 去target_domain的val_loader做测试

    # dataloader: 2.source domain: parameters for model_build
    # source_domain的model参数，只用来构建model
    num_classes, camera_num, view_num = get_parameter(cfg)

    # dataloader: 1.support_set :imgs & labels
    # support set读入
    train_loader_stage2 = make_dataloader_stage3(cfg)

    # dataloader: 3.target domain: val_loader for mAP & rk1
    # val_loader， num_query(for map%rk1)用target domain的
    ee, ff, val_loader, num_query, bb, cc, dd = make_stage3_testloader(cfg) # num_query:2228 images_num of target domain's query

    del bb, cc, dd, ee, ff

    # load param from stage1
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_param(cfg.DATASETS.WEIGHT)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(cfg, model, center_criterion)
    # WarmupMultiStepLR 是一种学习率调度策略
    # 在预热阶段结束后，根据预设的步骤 STEPS 来降低学习率，降低的比例由 GAMMA 控制。
    # 预热阶段的长度由 WARMUP_ITERS 控制，
    # 预热阶段的学习率调整策略由 WARMUP_METHOD 控制，
    # 预热阶段的学习率调整因子由 WARMUP_FACTOR 控制。
    # 这些都是从配置 cfg 中获取的。这种策略可以帮助模型在训练初期更好地收敛。
    scheduler_2stage = WarmupMultiStepLR(optimizer_2stage, cfg.SOLVER.STAGE2.STEPS, cfg.SOLVER.STAGE2.GAMMA, cfg.SOLVER.STAGE2.WARMUP_FACTOR,
                                  cfg.SOLVER.STAGE2.WARMUP_ITERS, cfg.SOLVER.STAGE2.WARMUP_METHOD)

    do_train_stage2(
        cfg,
        model,
        center_criterion,
        train_loader_stage2,
        val_loader,
        optimizer_2stage,
        optimizer_center_2stage,
        scheduler_2stage,
        loss_func,
        num_query, args.local_rank
    )







