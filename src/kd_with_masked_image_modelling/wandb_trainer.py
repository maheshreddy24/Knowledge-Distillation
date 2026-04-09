from models import StudentClassifier, TeacherClassifier, generator_network
from dataset_loader import DatasetImagenet
from optimisation import Optimisation
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wandb
from icecream import ic
import numpy as np


def setup():
    student = StudentClassifier(num_classes=1000)
    teacher = TeacherClassifier(num_classes=1000)
    generator = generator_network(d_model=1024)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    train_dataset = DatasetImagenet(root_dir='/media/system/ZERBUIS_EXT_STOR/dynamic_slam/imagenet', split='ILSVRC2012_img_train')
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True) # total of 12 million sampels

    test_dataset = DatasetImagenet(root_dir='/media/system/ZERBUIS_EXT_STOR/dynamic_slam/imagenet', split='ILSVRC2012_img_val')
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False) # 50k samples

    return student, teacher, generator, device, train_dataloader, test_dataloader 


sweep_configuration = {
    "method": "random",
    "parameters": {
        "epoch": {
            "values": [15, 20, 30]
        },
        "lr": {
            "values": list(np.linspace(1e-4, 1e-2, num=1000))
        },
        "weight_decay": {
            "values": list(np.linspace(1e-4, 1e-2, num=1000))
        }

    }
}

