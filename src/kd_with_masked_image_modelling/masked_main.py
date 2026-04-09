from dataset_loader import MaskedDataset
from vit_tiny import ViT_tiny_masked, vit_tiny_masked_wrapper
from torch.utils.data import DataLoader
from optimisation import SIMM_Optim
from argparse import ArgumentParser
from utils import get_logger
import os
import torch

def main(args):
    checkpoint = torch.load(args.ckpt_path)
    vit_tiny = ViT_tiny_masked()
    model = vit_tiny_masked_wrapper(vit_tiny)
    """
        Missing key(s) in state_dict: "model.mask_token", "model.pred_head.weight", "model.pred_head.bias". 
        Unexpected key(s) in state_dict: "mlp.0.weight", "mlp.0.bias", "mlp.3.weight", "mlp.3.bias". 
    """
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    
    train_dataset = MaskedDataset(split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=212, shuffle=True, num_workers=8)

    test_dataset = MaskedDataset(split='val')
    test_dataloader = DataLoader(test_dataset, batch_size=128, num_workers=8)

    epochs = 100
    device = 'cuda'
    learning_rate = 5e-4
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)
    optimiser = SIMM_Optim(model, device, epochs, learning_rate, train_dataloader, test_dataloader, output_dir)
    optimiser.train()
    # optimiser.load_checkpoint(args.ckpt_path)
    # optimiser.evaluation()

if __name__ == "__main__":
    parser = ArgumentParser(description='Training')
    parser.add_argument('--output_dir', type=str, required=True)
    # parser.add_argument('--logger_name', type=str, required=True)
    parser.add_argument('--resume', type=bool, required=True)
    parser.add_argument('--ckpt_path', type=str, required=False, default="/media/system/ZERBUIS_EXT_STOR/dynamic_slam/src/vit_tiny_checkpoints/model_epoch_76.pth")
    args = parser.parse_args()
    
    main(args)