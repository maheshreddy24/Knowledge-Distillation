
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from datetime import datetime
from transformers import get_cosine_schedule_with_warmup
import torch.nn.functional as F
from icecream import ic
import time
from utils import get_logger
import wandb

class Optimisation():
    def __init__(self, student, teacher, generator, device, epochs, learning_rate, train_dataloader, test_dataloader, temperature , output_dir, train_batch_size):
        # models
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        # ! this will directly return the loss
        self.generator = generator.to(device)


        self.device = device
        self.lr = learning_rate
        self.epochs = epochs
        self.loss_list = []
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader


        # this is for the main classification loss for the student
        self.criterion = nn.CrossEntropyLoss()
        
        # batchmean is the average loss over the batch, which is what we want for distillationq
        self.soft_loss = nn.KLDivLoss(reduction="batchmean")
        self.temperature = temperature

        # the ViTKD uses lambda as 1        
        # # this is the mse for the classification
        # self.lambda_ = 0.5
        # this is for the soft loss
        self.alpha = 0.5
        # this is for the feature generation loss, if u want learn features very similar to the distribution of dinov2 then increase this
        # self.beta = 5e-4
        self.start_epoch = 0

        WEIGHT_DECAY = 0.05
        WARMUP_RATIO = 0.05
        self.optimizer = optim.AdamW(self.student.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY)
        num_training_steps = self.epochs * len(train_dataloader)
        num_warmup_steps = int(WARMUP_RATIO * num_training_steps)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        self.d_model = 1024
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(self.output_dir, current_date)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger = get_logger(os.path.join(self.checkpoint_dir, 'logger.log'))

        wandb.init(
            project = f"{self.output_dir}_{current_date}",
            name = 'logger',
            mode = "online",
            config = {
                "learning_rate": self.lr,
                "epochs": self.epochs,
                "batch_size": train_batch_size,
                "weight_decay": WEIGHT_DECAY,
                "warmup_ratio": WARMUP_RATIO,
                "model": "vit_tiny",
                "dataset": "ImageNet"
            }
        )

    def save_checkpoint(self, epoch):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss_list': self.loss_list,
            'epoch': epoch,
        }, checkpoint_path)
        self.logger.info(f"Checkpoint saved at: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.student.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.loss_list = checkpoint['loss_list']
        self.start_epoch = checkpoint['epoch']
        self.logger.info(f"Checkpoint loaded from: {checkpoint_path}, epoch: {checkpoint['epoch']}")
    

    def resume_training(self, checkpoint_path):
        self.logger.info("Resuming training")
        self.load_checkpoint(checkpoint_path)

    
    def evaluation(self, epoch):
        self.student.eval()
        # self.teacher.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(self.test_dataloader, total=len(self.test_dataloader)):
                images = images.to(self.device)
                labels = labels.to(self.device)

                    # teacher_output, _ = self.teacher(images)
                student_output, _ = self.student(images)
                
                _, predicted = torch.max(student_output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = (correct / total) * 100
            self.logger.info(f"Test Accuracy: {accuracy:.4f}")
            wandb.log(
                {
                    'test_epoch': epoch,
                     'test_accuracy': accuracy
                }
            )

        
    def train(self):
        for epoch in tqdm(range(self.start_epoch, self.epochs + 1), total=len(self.epochs + 1)):
            self.student.train()
            self.teacher.eval()  # Teacher is in eval mode
            batch_losses = []
            self.logger.info(f'\nEpoch {epoch}/{self.epochs}')

            for images, labels in tqdm(self.train_dataloader, total=len(self.train_dataloader), unit="batch"):
                input_batch = images.to(self.device)
                output_batch = labels.to(self.device)
                self.optimizer.zero_grad()

                # the student_feature_dim = 192
                # the teacher_featre_dim = 1024

                with torch.no_grad():
                    # this will return the logits & the features
                    teacher_output, teacher_features = self.teacher(input_batch)

                student_output, student_features = self.student(input_batch)

                
                # the output will be the logits from the linear layer
                soft_teacher = F.softmax(teacher_output / self.temperature, dim=-1)
                soft_student = F.log_softmax(student_output / self.temperature, dim=-1)

                distillation_loss = (
                    # this loss is KLD
                    self.soft_loss(soft_student, soft_teacher)
                    * (self.temperature ** 2)
                )

                # ! this is commented for now
                
                std_feat = student_features.permute(0, 2, 1) # bs, 192, 257
                std_feat = std_feat[:, :, 1:] # bs, 192, 256
                bs, ch, emb = std_feat.shape
                # the emb/16 should be w/patch_size, h/patch_size
                # the .contiguous() will return a single block of memory for the whole tensor which will be more optimal for view operations, 
                # if the tensor is already contiguous then no changes
                std_feat = std_feat.contiguous().view(bs, ch, 16, 16)

                teach_feat = teacher_features.permute(0, 2, 1) #bs, 1024, 257
                teach_feat = teach_feat[:, :, 1:] # bs, 1024, 256
                bs, ch, emb = teach_feat.shape
                teach_feat = teach_feat.contiguous().view(bs, ch, 16, 16)
                
                feature_loss = self.generator(std_feat, teach_feat)

                # the self.criterion is the cross entropy loss for the student output and the true labels
                hard_ce = self.criterion(student_output, output_batch)
                loss = hard_ce + \
                       self.alpha * distillation_loss + \
                       feature_loss
                
                # logging
                wandb.log({
                    'hard_ce': hard_ce,
                    'soft_ce': distillation_loss,
                    'feature_loss': feature_loss,
                    'total_loss': loss,
                })

                # Backpropagation
                loss.backward()
                self.optimizer.step()            
                self.scheduler.step()
                batch_losses.append(loss.item())

            # Average loss calculation
            epoch_loss = sum(batch_losses) / len(batch_losses)
            self.loss_list.append(epoch_loss)
            
            wandb.log(
                {
                    'epoch': epoch,
                    'average_loss': epoch_loss
                }
            )
            self.logger.info(f"Epoch {epoch}, Average Loss: {epoch_loss:.4f}")

            # Save and evaluate every 10 epochs
            self.save_checkpoint(epoch)
            try:
                self.evaluation(epoch)
            except Exception as err:
                print(err)
        self.save_checkpoint(self.epochs)
