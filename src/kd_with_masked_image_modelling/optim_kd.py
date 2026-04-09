import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from datetime import datetime
from transformers import get_cosine_schedule_with_warmup
import torch.nn.functional as F
from icecream import ic
from utils import get_logger



class SIMPLE_KD_Optimisation():
    def __init__(self, student, teacher, generator, device, epochs, learning_rate, train_dataloader, test_dataloader, temperature, output_dir):
        # models
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        # this is not required if using masked image modeeling in KD, 
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
        # this is for the feature generation loss
        self.beta = 1e-3

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
        # this is the embd dim of dinov2 large
        self.d_model = 1024
        self.output_dir = output_dir
        current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.checkpoint_dir = os.path.join(self.output_dir, current_date)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # the logger txt
        self.logger = get_logger(os.path.join(self.checkpoint_dir, 'logger' + '.txt'))
        self.logger.info("Hello!!")


    def patchify(self, images, patch_size=14):
        B, C, H, W = images.shape

        h = H // patch_size
        w = W // patch_size

        x = images.reshape(B, C, h, patch_size, w, patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, h*w, patch_size*patch_size*C)

        return x
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_list': self.loss_list,
        }, checkpoint_path)
        print(f"Checkpoint saved at: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.student.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_list = checkpoint['loss_list']
        print(f"Checkpoint loaded from: {checkpoint_path}, epoch: {checkpoint['epoch']}")
    
    def feature_generation_loss(self, student_features, teacher_features, mask_lambda=0.5):
        """
        student_features: [B, N, D]
        teacher_features: [B, N, D]
        """
        B, N, D = student_features.shape
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.mask_token = self.mask_token.to(self.device)


        # 1. Create random mask (token-level)
        mask = torch.rand(B, N, device=student_features.device) < mask_lambda  # [B, N]

        # 2. Replace masked tokens with learnable mask token
        mask_expanded = mask.unsqueeze(-1)  # [B, N, 1]
        mask_expanded = mask_expanded.to(self.device)
        masked_student = student_features.clone()
        masked_student = masked_student.to(self.device)
        masked_student = torch.where(
            mask_expanded,
            self.mask_token.expand(B, N, D),
            masked_student
        )

        # 3. Pass through generator
        generated = self.generator(masked_student)  # [B, N, D]

        # 4. Compute MSE only on masked tokens
        loss = F.mse_loss(
            generated[mask],
            teacher_features[mask]
        )

        return loss

    def evaluation(self):
        self.student.eval()
        # self.teacher.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(self.test_dataloader, total=len(self.test_dataloader)):
                images = images.to(self.device)
                labels = labels.to(self.device)

                #! logits, projected
                student_output, _ = self.student(images, mask_req=False)
                
                _, predicted = torch.max(student_output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = (correct / total) * 100
            print(f"Test Accuracy: {accuracy:.4f}")

        
    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.student.train()
            self.teacher.eval()  # Teacher is in eval mode
            batch_losses = []
            print(f'\nEpoch {epoch}/{self.epochs}')

            for images, labels in tqdm(self.train_dataloader, total=len(self.train_dataloader), unit="batch"):
                input_batch = images.to(self.device)
                output_batch = labels.to(self.device)

                with torch.no_grad():
                    # this will return the logits & the features
                    teacher_output, teacher_features = self.teacher(input_batch)
                
                # ! student_features: [B, N, 1024], student_ouput: [bs, Num_classes], mask: [B, N]
                student_output, student_features, mask = self.student(input_batch, mask_req=True)

                # the output will be the logits from the linear layer
                soft_teacher = F.softmax(teacher_output / self.temperature, dim=-1)
                soft_student = F.log_softmax(student_output / self.temperature, dim=-1)

                distillation_loss = (
                    # this loss is KLD
                    self.soft_loss(soft_student, soft_teacher)
                    * (self.temperature ** 2)
                )

                # the masked features are projected into the teacher dim
                masked_loss = torch.abs(teacher_features - student_features)
                masked_loss = (masked_loss.mean(dim=-1) * mask).sum() / mask.sum()
                # ! better do this
                # masked_loss = F.mse_loss(
                #     student_features[mask],
                #     teacher_features[mask]
                # )

                # ! this is commented for now
                # feature_loss = self.feature_generation_loss(student_features, teacher_features)

                # the self.criterion is the cross entropy loss for the student output and the true labels
                loss = self.criterion(student_output, output_batch) + \
                       self.alpha * distillation_loss + \
                       self.beta * masked_loss

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                

                batch_losses.append(loss.item())

            # Average loss calculation
            epoch_loss = sum(batch_losses) / len(batch_losses)
            self.loss_list.append(epoch_loss)
            print(f"Epoch {epoch}, Average Loss: {epoch_loss:.4f}")

            # Scheduler step
            self.scheduler.step()

            # Save and evaluate every 10 epochs
            if epoch % 5 == 0 or epoch < 10:
                self.save_checkpoint(epoch)
                self.evaluation()

        self.save_checkpoint(self.epochs)
        # return output
    
