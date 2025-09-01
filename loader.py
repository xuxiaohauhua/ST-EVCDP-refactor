import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional

# def read_dataset():
#     occ = pd.read_csv('datasets/occupancy.csv', index_col=0, header=0)
#     inf = pd.read_csv('datasets/information.csv', index_col=None, header=0)
#     prc = pd.read_csv('datasets/price.csv', index_col=0, header=0)
#     adj = pd.read_csv('datasets/adj.csv', index_col=0, header=0)  # check
#     dis = pd.read_csv('datasets/distance.csv', index_col=0, header=0)
#     time = pd.read_csv('datasets/time.csv', index_col=None, header=0)

#     col = occ.columns
#     cap = np.array(inf['count'], dtype=float).reshape(1, -1)  # parking_capability
#     occ = np.array(occ, dtype=float) / cap
#     prc = np.array(prc, dtype=float)
#     adj = np.array(adj, dtype=float)
#     dis = np.array(dis, dtype=float)
#     time = pd.to_datetime(time, dayfirst=True)
#     return occ, prc, adj, col, dis, cap, time, inf

# def create_rnn_data(dataset, lookback, predict_time):
#     x = []
#     y = []
#     for i in range(len(dataset) - lookback - predict_time):
#         x.append(dataset[i:i + lookback])
#         y.append(dataset[i + lookback + predict_time - 1])
#     return np.array(x), np.array(y)

# def get_a_delta(adj):  # D^-1/2 * A * D^-1/2
#     # adj.shape = np.size(node, node)
#     deg = np.sum(adj, axis=0)
#     deg = np.diag(deg)
#     deg_delta = np.linalg.inv(np.sqrt(deg))
#     a_delta = np.matmul(np.matmul(deg_delta, adj), deg_delta)
#     return a_delta


# def division(data, train_rate, valid_rate, test_rate):
#     data_length = len(data)
#     train_division_index = int(data_length * train_rate)
#     valid_division_index = int(data_length * (train_rate + valid_rate))
#     test_division_index = int(data_length * (1 - test_rate))
#     train_data = data[:train_division_index, :]
#     valid_data = data[train_division_index:valid_division_index, :]
#     test_data = data[test_division_index:, :]
#     return train_data, valid_data, test_data


# def set_seed(seed, flag):
#     if flag == True:
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)


class CSVTimeSeriesDataset(Dataset):
    """CSV时序数据集"""
    
    def __init__(self, csv_file: str, feature_columns: list, target_column: str,
                 seq_length: int, pred_length: int, scaler_type: str = 'standard'):
        """
        Args:
            csv_file: CSV文件路径
            feature_columns: 特征列名列表
            target_column: 目标列名
            seq_length: 输入序列长度
            pred_length: 预测序列长度
            scaler_type: 标准化类型 ('standard' 或 'minmax')
        """
        self.seq_length = seq_length
        self.pred_length = pred_length
        
        # 读取数据
        self.df = pd.read_csv(csv_file)
        
        # 提取特征和目标
        self.features = self.df[feature_columns].values
        self.targets = self.df[target_column].values.reshape(-1, 1)
        
        # 数据标准化
        if scaler_type == 'standard':
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        else:
            self.feature_scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()
            
        self.features = self.feature_scaler.fit_transform(self.features)
        self.targets = self.target_scaler.fit_transform(self.targets)
        
        # 创建序列
        self.sequences = []
        self.labels = []
        
        for i in range(len(self.features) - seq_length - pred_length + 1):
            # 输入序列 (包含特征和目标的历史值)
            seq_features = self.features[i:i+seq_length]
            seq_targets = self.targets[i:i+seq_length]
            seq = np.concatenate([seq_features, seq_targets], axis=1)
            
            # 预测目标
            label = self.targets[i+seq_length:i+seq_length+pred_length]
            
            self.sequences.append(seq)
            self.labels.append(label.flatten())
            
        self.sequences = np.array(self.sequences)
        self.labels = np.array(self.labels)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.labels[idx])
        )
    
    def inverse_transform_target(self, scaled_data):
        """反标准化目标数据"""
        return self.target_scaler.inverse_transform(scaled_data.reshape(-1, 1)).flatten()


def create_dataloaders(csv_file: str, feature_columns: list, target_column: str,
                      seq_length: int, pred_length: int, batch_size: int,
                      train_ratio: float = 0.7, val_ratio: float = 0.15,
                      scaler_type: str = 'standard') -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建训练、验证、测试数据加载器"""
    
    # 读取数据并按时间顺序分割
    df = pd.read_csv(csv_file)
    total_len = len(df)
    
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))
    
    # 分割数据
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    # 保存到临时文件
    train_file = 'temp_train.csv'
    val_file = 'temp_val.csv'
    test_file = 'temp_test.csv'
    
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    # 创建数据集
    train_dataset = CSVTimeSeriesDataset(train_file, feature_columns, target_column,
                                        seq_length, pred_length, scaler_type)
    val_dataset = CSVTimeSeriesDataset(val_file, feature_columns, target_column,
                                      seq_length, pred_length, scaler_type)
    test_dataset = CSVTimeSeriesDataset(test_file, feature_columns, target_column,
                                       seq_length, pred_length, scaler_type)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_dataset.target_scaler