import torch
import torch.nn as nn
import torch.nn.functional as F
import functions as fn
import copy
import math

use_cuda = True
device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
fn.set_seed(seed=2023, flag=True)

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TimeSeriesTransformer(nn.Module):
    """时序预测Transformer模型"""
    
    def __init__(self, input_dim, model_dim, num_heads, num_layers, 
                 seq_length, pred_length, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.seq_length = seq_length
        self.pred_length = pred_length
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, model_dim)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(model_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, input_dim * pred_length)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length, input_dim)
        Returns:
            output: (batch_size, pred_length, input_dim)
        """
        batch_size = x.size(0)
        
        # 输入嵌入
        x = self.input_embedding(x) * math.sqrt(self.model_dim)
        
        # 调整维度: (seq_length, batch_size, model_dim)
        x = x.permute(1, 0, 2)
        
        # 位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer编码
        encoded = self.transformer_encoder(x)
        
        # 使用最后一个时间步的输出进行预测
        last_hidden = encoded[-1]  # (batch_size, model_dim)
        
        # 预测
        prediction = self.prediction_head(last_hidden)
        
        # 重塑输出: (batch_size, pred_length, input_dim)
        prediction = prediction.view(batch_size, self.pred_length, self.input_dim)
        
        return prediction