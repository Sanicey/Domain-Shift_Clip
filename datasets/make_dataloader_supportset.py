import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from PIL import Image
import os
from torchvision.transforms import Compose, Resize, ToTensor
import random

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    # 整理训练数据批次，返回图像、人物ID（pids）、摄像机ID（camids）和视图ID（viewids）
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

def val_collate_fn(batch):
    # 整理验证数据批次，返回图像、pids、camids、camids_batch（可能用于不同的用途）、viewids和图像路径
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
}

def make_dataloader_supportset1(cfg):

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    return num_classes, cam_num, view_num

def make_dataloader_supportset2(cfg, num_of_splits = 100):
    target_domain_dataset = __factory[cfg.Domain_Shift_DATASETS.NAMES](root=cfg.Domain_Shift_DATASETS.ROOT_DIR)

    # 修改预处理函数，以适应多张图片
    preprocess = Compose([
        Resize((256,128)),
        ToTensor()
    ])

    # target_domain_dataset.traindir 目标图片所在文件夹地址
    # 有个问题，读入的是原始数据，没有transform
    train_dir = target_domain_dataset.train_dir
    # image_paths = [os.path.join(train_dir, img) for img in os.listdir(train_dir)]
    image_paths = [os.path.join(train_dir, img) for img in os.listdir(train_dir)]
    images = [Image.open(img_path) for img_path in image_paths]

    # 打乱图片顺序
    random.shuffle(images)

    # 将图片分割成等大小的列表
    split_images = []
    split_size = len(images) // num_of_splits
    remainder = len(images) % num_of_splits

    start = 0
    for i in range(num_of_splits):
        end = start + split_size
        if i < remainder:
            end += 1
        split_images.append(images[start:end])
        start = end

    # # 预处理图片并拼接在一起
    # target_domain_all_images = torch.stack([preprocess(img)for img in images])

    # 预处理图片，每一组内部拼接在一起
    target_domain_all_images = [torch.stack([preprocess(img) for img in img_list]) for img_list in split_images]

    return target_domain_all_images, images
