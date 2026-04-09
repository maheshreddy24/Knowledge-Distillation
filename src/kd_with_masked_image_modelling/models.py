import torch
from torch.utils.data import Dataset
from torchvision import datasets
from transformers import AutoModel
import torch.nn as nn
import os
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm
from icecream import ic
from PIL import Image
import random
import numpy as np
from torchvision.datasets import OxfordIIITPet
import timm
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from icecream import ic
from vit_tiny import vit_tiny_classifier

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)



class TeacherWrapper(nn.Module):
    def __init__(self, model_name = 'facebook/dinov2-large'):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode

    def forward(self, x):
        with torch.no_grad():
            params =  self.model(x).last_hidden_state  # [B, 197, 1024]
        return params
    
#! dinov2 large has 257 patch size
class TeacherClassifier(nn.Module):
    def __init__(self, num_classes = 1000, model_name = 'facebook/dinov2-large'):
        super().__init__()

        self.teacher = TeacherWrapper(model_name)
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        teacher_features = self.teacher(x) # bs, 257, 1024
        class_token = teacher_features[:, 0, :] # bs, 1024
        out = self.mlp(class_token) 
        return out, teacher_features

class StudentWrapper(nn.Module):
    def __init__(self, student):
        super().__init__()
        self.student = student

    def forward(self, x):
        student_out = self.student(x)   # [B, 257, 192]
        return student_out


# ! if you want to use this for a different dataset with different class size u can use another mlp to project from this, rather than 
class StudentClassifier(nn.Module):
    def __init__(self, num_classes = 1000):
        super().__init__()

        # ! this is the pretrained vit tiny, with the classifier but we are not using the classifier outputs
        self.model_tiny = vit_tiny_classifier(n_classes=1000)
        # ! freezing the vit tiny, the checkpoint used has
        self.load_checkpoint_student()
        # this will return bs, 257, 192  
        self.student = StudentWrapper(self.model_tiny)
        self.student_dim = 192

        self.mlp = nn.Sequential(
            nn.Linear(self.student_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # this returns the same tensor, from the vit tiny
        projected_features = self.student(x) # bs, 257,192
        class_token = projected_features[:, 0, :] # bs, 192
        out = self.mlp(class_token) # bs, num_classes

        return out, projected_features
    
    def load_checkpoint_student(self):
        checkpoint = torch.load("/media/system/ZERBUIS_EXT_STOR/dynamic_slam/src/vit_tiny_checkpoints/model_epoch_76.pth")
        self.model_tiny.load_state_dict(checkpoint['model_state_dict'])

    def freeze_student(self):
        for param in self.student.parameters():
            param.requires_grad = False 
    
    def unfreeze_student(self): 
        for param in self.student.parameters():
            param.requires_grad = True


# class generator_network(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
    
#         self.generator = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.GELU(),
#             nn.Linear(d_model, d_model)
#         )
    
#     def forward(self, x):
#         return self.generator(x)




class MGDLoss(nn.Module):

    """PyTorch version of `Masked Generative Distillation`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00007
        lambda_mgd (float, optional): masked ratio. Defaults to 0.5
    """
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 alpha_mgd=0.0007,
                 lambda_mgd=0.15,
                 ):
        super(MGDLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
    
        if student_channels != teacher_channels:
            # if the dim of student channels is different from the teacher channel then it would increase the number of channels.
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None


        # the ouput will be the same after 
        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))


    def forward(self,
                preds_S,
                preds_T):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        if self.align is not None:
            preds_S = self.align(preds_S)
    
        loss = self.get_dis_loss(preds_S, preds_T)*self.alpha_mgd
            
        return loss

    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N,C,1,1)).to(device)
        # mat = torch.rand((N,1,H,W)).to(device)
        # ! if the condition is true the val is 0 else 1.
        mat = torch.where(mat < self.lambda_mgd, 0, 1).to(device)

        # ! so here if the pred_S has grad True, and if the the channel has to be masked then it will be 0
        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation(masked_fea)

        dis_loss = loss_mse(new_fea, preds_T)/N

        return dis_loss