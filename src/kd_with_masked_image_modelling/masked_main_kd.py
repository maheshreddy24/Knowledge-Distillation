from models import TeacherModel, generator_network, StudentClassifier
from vit_tiny import vit_tiny_masked_wrapper 
from dataset_loader import DatasetImagenet, OxfordPetDataset
from optimisation import SIMPLE_KD_Optimisation
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from argparse import ArgumentParser


def main(args):

    
    # this will output logits, features (bs, N, 1024)
    teacher = TeacherModel(num_classes=1000)
    generator = generator_network(d_model=1024)
    # this is a wrapper around the the masked vit pass mask_req 
    """
        Missing key(s) in state_dict: "model.mask_token", "model.pred_head.weight", "model.pred_head.bias". 
        Unexpected key(s) in state_dict: "mlp.0.weight", "mlp.0.bias", "mlp.3.weight", "mlp.3.bias". 
    """
    vit_tiny = vit_tiny_masked_wrapper()
    # this takes in the whole student model with pretrained vit tiny,
    student = StudentClassifier(vit_tiny, num_classes=1000, checkpoint_path="/media/system/ZERBUIS_EXT_STOR/dynamic_slam/src/vit_tiny_checkpoints/model_epoch_76.pth")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    # ! Imagenet
    train_dataset = DatasetImagenet(root_dir='/media/system/ZERBUIS_EXT_STOR/dynamic_slam/imagenet', split='ILSVRC2012_img_train')
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True) # total of 12 million sampels

    test_dataset = DatasetImagenet(root_dir='/media/system/ZERBUIS_EXT_STOR/dynamic_slam/imagenet', split='ILSVRC2012_img_val')
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False) # 50k samples
    
    #! oxford pet dataset
    # train_dataset = OxfordPetDataset(root_dir="/media/system/ZERBUIS_EXT_STOR/dynamic_slam/experiments/data/oxford-iiit-pet", split="trainval")
    # train_dataloader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=64,      
    #     shuffle=True)

    # test_dataset = OxfordPetDataset(root_dir="/media/system/ZERBUIS_EXT_STOR/dynamic_slam/experiments/data/oxford-iiit-pet", split="test")
    # test_dataloader = DataLoader(
    #     dataset=test_dataset,
    #     batch_size=32,      
    #     shuffle=False)


    epochs = 20
    optimiser = SIMPLE_KD_Optimisation(student=student, teacher=teacher, generator=generator, device=device, epochs=epochs, train_dataloader=train_dataloader, test_dataloader=test_dataloader, \
                                       temperature=4, output_dir=args.output_dir, learning_rate=5e-4)
    optimiser.train()

if __name__ == "__main__":
    parser = ArgumentParser(description='Training')
    parser.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()
    main(args)