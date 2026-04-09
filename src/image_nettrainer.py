import torch
from torch.utils.data import Dataset
from torchvision import datasets
from transformers import AutoImageProcessor, AutoModel
import torch.nn as nn
import os
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import wandb
from tqdm import tqdm
from icecream import ic
from transformers import get_cosine_schedule_with_warmup
from PIL import Image
import random
import numpy as np
import timm
import cv2
from vit_tiny_main import vit_tiny_classifier
from utils import get_logger
from icecream import ic
from scipy.io import loadmat
from PIL import Image
import scipy.io as sio
from argparse import ArgumentParser, Namespace
from datetime import datetime
import logging
from pathlib import Path

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)



class DatasetImagenet(Dataset):
    def __init__(self, root_dir, split="train"):
        """
        root_dir/
            ILSVRC2012_img_train/
            ILSVRC2012_img_val/
            ILSVRC2012_validation_ground_truth.txt
            meta.mat
        """

        self.root_dir = root_dir
        self.split = split

        meta = loadmat(
            os.path.join(root_dir, "meta.mat"),
            struct_as_record=False,
            squeeze_me=True
        )

        synsets = meta["synsets"]

        # Keep only low-level synsets (ILSVRC2012_ID <= 1000)
        id_to_wnid = {}
        for s in synsets:
            ilsvrc_id = int(s.ILSVRC2012_ID)
            if ilsvrc_id <= 1000:
                id_to_wnid[ilsvrc_id - 1] = s.WNID  # zero-based

        # Sort by official ID order
        self.wnids = [id_to_wnid[i] for i in range(1000)]

        # Internal class mapping
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}
        # ic(len(self.class_to_idx))
        # ic(self.class_to_idx)


        if split == "train":
            self.transform = transforms.Compose([
                # (C, H, W) 
                transforms.RandomResizedCrop((336, 336), scale=(0.08, 1.0)),
                # transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(336),
                transforms.CenterCrop(336),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

        self.samples = []

        if split == "train":
            train_dir = os.path.join(root_dir, "ILSVRC2012_img_train")

            for wnid in self.wnids:
                cls_path = os.path.join(train_dir, wnid)
                if not os.path.isdir(cls_path):
                    continue

                for img_name in os.listdir(cls_path):
                    img_path = os.path.join(cls_path, img_name)
                    label = self.class_to_idx[wnid]
                    self.samples.append((img_path, label))
        else:
            val_dir = os.path.join(root_dir, "ILSVRC2012_img_val")
            gt_file = os.path.join(root_dir, "ILSVRC2012_validation_ground_truth.txt")

            with open(gt_file) as f:
                official_ids = [int(x.strip()) - 1 for x in f.readlines()]

            img_files = sorted(os.listdir(val_dir))

            for img_name, official_id in zip(img_files, official_ids):
                wnid = id_to_wnid[official_id]
                label = self.class_to_idx[wnid]

                img_path = os.path.join(val_dir, img_name)
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label
    

class Trainer():
    def __init__(self, logger_name, resume, best_ckpt, output_dir):

        train_bs = 64
        train_dataset = DatasetImagenet(root_dir='/media/system/ZERBUIS_EXT_STOR/dynamic_slam/imagenet', split='train')
        self.train_dataloader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=16) # total of 12 million sampels

        test_dataset = DatasetImagenet(root_dir='/media/system/ZERBUIS_EXT_STOR/dynamic_slam/imagenet', split='test')
        self.test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False) # 50k samples
        self.model = vit_tiny_classifier(img_size=(336, 336), n_classes=1000)
        self.model = self.model.to('cuda')


        self.resume_training_cond = resume
        self.checkpoint_path = best_ckpt


        LR = 5e-4/4
        self.EPOCHS = 300

        WEIGHT_DECAY = 0.05/4
        WARMUP_RATIO = 0.05*6

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )

        self.device = 'cuda'
        self.model = self.model.to(self.device)

        num_training_steps = self.EPOCHS * len(self.train_dataloader)
        num_warmup_steps = int(WARMUP_RATIO * num_training_steps)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.output_dir = output_dir
        wandb.init(
            project = f"{self.output_dir}",
            name = logger_name,
            mode = "online",
            config = {
                "learning_rate": LR,
                "epochs": self.EPOCHS,
                "batch_size": 256,
                "weight_decay": WEIGHT_DECAY,
                "warmup_ratio": WARMUP_RATIO,
                "model": "vit_tiny",
                "dataset": "ImageNet"
            }
        )
        os.makedirs(self.output_dir, exist_ok=True)
        abs_path = os.path.abspath(self.output_dir)
        print(abs_path)
        self.logger = get_logger("/media/system/ZERBUIS_EXT_STOR/dynamic_slam/src/" + self.output_dir + f'/{logger_name}.txt')
        self.logger.info("Logger test message")


    def resume_training(self, checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model
        self.model.load_state_dict(checkpoint["model_state_dict"], strict = True)

        # Load optimizer if provided
        if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # # Load scheduler if provided
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1

        print(f"Resumed from epoch {checkpoint['epoch']}")
        return start_epoch

    def save_checkpoint(self, epoch):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, f'model_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }, checkpoint_path)
        print(f"Checkpoint saved at: {checkpoint_path}")


    def evaluate(self, epoch):
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in tqdm(self.test_dataloader, desc='Evaulating', total=len(self.test_dataloader)):

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * (val_correct / val_total)
        
        self.logger.info(f"Epoch [{epoch}/{self.EPOCHS}]")
        self.logger.info(f"Val Acc: {val_acc:.2f}%")
        wandb.log({
            "val_accuracy": val_acc,
            "epoch": epoch
        })
        

    def train(self):
        start_epoch = 0
        ic(self.resume_training_cond)
        if self.resume_training_cond:
            ic(self.checkpoint_path)
            start_epoch = self.resume_training(self.checkpoint_path)
            self.logger.info("Training resumed")
            self.evaluate(start_epoch)
            
        self.logger.info(f'Training started: {start_epoch}')
        for epoch in tqdm(range(start_epoch, self.EPOCHS), desc='Training'):
            self.model.train()
            total_loss = 0
            correct, total = 0, 0
            self.logger.info("=========================")
            self.logger.info(f"{epoch}/{self.EPOCHS} training")

            for images, labels in tqdm(self.train_dataloader, desc='Batches', total=len(self.train_dataloader)):

                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(images)           # (B, num_classes)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()

                # accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = 100 * (correct / total)
            avg_loss = total_loss / len(self.train_dataloader)
            self.logger.info(f'Training accuracy: {train_acc}')
            self.logger.info(f"Average loss: {avg_loss}")
            wandb.log({
                "train_loss": avg_loss,
                "train_accuracy": train_acc,
                "epoch": epoch,
                "learning_rate": self.optimizer.param_groups[0]["lr"]
            })
            self.save_checkpoint(epoch)
            # eval every epoch
            self.evaluate(epoch)

            # if epoch < 10 or epoch % 10 == 5:
            #     self.evaluate(epoch)


if __name__ == "__main__":
    
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--resume", type=bool, required=False, default=False)
    parser.add_argument("--model_path", required=False, type=str)
    parser.add_argument("--logger_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    instance = Trainer(resume=args.resume, best_ckpt=args.model_path, logger_name = args.logger_name, output_dir = args.output_dir)
    instance.train()