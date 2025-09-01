import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
from typing import Dict, List, Tuple

from models import TimeSeriesTransformer


class TransformerTrainer:
    """Transformer模型训练器"""
    
    def __init__(self, model: TimeSeriesTransformer, config: dict, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.0001)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 记录训练过程
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # TensorBoard
        self.writer = SummaryWriter(f"runs/{config['experiment_name']}")
        
    def train_epoch(self, train_loader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (sequences, targets) in enumerate(pbar):
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(sequences)
            
            # 计算损失
            # predictions: (batch_size, pred_length, input_dim)
            # targets: (batch_size, pred_length)
            # 只使用目标列进行损失计算
            pred_targets = predictions[:, :, -1]  # 假设目标是最后一列
            loss = self.criterion(pred_targets, targets)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
            
        return total_loss / num_batches
    
    def validate(self, val_loader) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(sequences)
                pred_targets = predictions[:, :, -1]
                loss = self.criterion(pred_targets, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches
    
    def train(self, train_loader, val_loader, num_epochs: int) -> Dict[str, List[float]]:
        """完整训练过程"""
        print(f"开始训练，总共 {num_epochs} 个epoch")
        
        for epoch in range(num_epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss = self.validate(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # TensorBoard记录
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, 'best_model.pth')
                
            # 打印进度
            print(f'Epoch [{epoch+1}/{num_epochs}] - '
                  f'Train Loss: {train_loss:.6f}, '
                  f'Val Loss: {val_loss:.6f}, '
                  f'LR: {self.optimizer.param_groups[0]["lr"]:.8f}')
            
            # 早停检查
            if self.config.get('early_stopping', False):
                if self._early_stopping_check(epoch):
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                    
        self.writer.close()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def save_checkpoint(self, epoch: int, filename: str):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        save_path = os.path.join(self.config['checkpoint_dir'], filename)
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f"模型已保存到: {save_path}")
    
    def load_checkpoint(self, filepath: str):
        """加载模型检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        return checkpoint['epoch']
    
    def _early_stopping_check(self, epoch: int) -> bool:
        """早停检查"""
        patience = self.config.get('early_stopping_patience', 20)
        if len(self.val_losses) < patience:
            return False
            
        recent_losses = self.val_losses[-patience:]
        return all(recent_losses[0] <= loss for loss in recent_losses[1:])