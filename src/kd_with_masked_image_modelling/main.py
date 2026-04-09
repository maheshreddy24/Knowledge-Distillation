from models import TeacherClassifier, MGDLoss, StudentClassifier
from dataset_loader import DatasetImagenet
from optimisation import Optimisation
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from icecream import ic
from argparse import ArgumentParser



def main(args):
    student = StudentClassifier()
    teacher = TeacherClassifier()

    generator = MGDLoss(student_channels=192, teacher_channels=1024)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = DatasetImagenet(root_dir='/media/system/ZERBUIS_EXT_STOR/dynamic_slam/imagenet', split='train')

    bs = 180
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=16) # total of 12 million sampels

    test_dataset = DatasetImagenet(root_dir='/media/system/ZERBUIS_EXT_STOR/dynamic_slam/imagenet', split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False) # 50k samples
    
    epochs = 100
    optimiser = Optimisation(student, teacher, generator, device, epochs, 5e-4, train_dataloader, test_dataloader, 4, output_dir=args.output_dir, train_batch_size = bs)
    ic(args.resume)
    if args.resume:
        print('Resuming training')
        optimiser.resume_training(args.ckpt_path)
    optimiser.train()

if __name__ == "__main__":
    parser = ArgumentParser(description='Training')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument("--resume", type=bool, required=False, default=False)
    parser.add_argument('--ckpt_path', type=str, required=False)
    args = parser.parse_args()
    main(args)