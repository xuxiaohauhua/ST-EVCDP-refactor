"""
评估指标计算
"""
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Union


def calculate_metrics(y_true: Union[np.ndarray, torch.Tensor], 
                     y_pred: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
    """
    计算时序预测的评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        Dict[str, float]: 包含各种评估指标的字典
    """
    # 转换为numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # 展平数组用于计算
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # 移除NaN值
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
    if len(y_true_clean) == 0:
        return {
            'MSE': np.nan,
            'RMSE': np.nan,
            'MAE': np.nan,
            'MAPE': np.nan,
            'R2': np.nan,
            'RAE': np.nan
        }
    
    # 计算各种指标
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-8))) * 100
    
    # R² (决定系数)
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    # RAE (Relative Absolute Error)
    rae = np.sum(np.abs(y_true_clean - y_pred_clean)) / (np.sum(np.abs(y_true_clean - np.mean(y_true_clean))) + 1e-8)
    
    return {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'MAPE': float(mape),
        'R2': float(r2),
        'RAE': float(rae)
    }


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算方向准确率 (预测趋势是否正确)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        float: 方向准确率 (0-1)
    """
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0
        
    true_diff = np.diff(y_true.flatten())
    pred_diff = np.diff(y_pred.flatten())
    
    correct_directions = (true_diff * pred_diff) >= 0
    return np.mean(correct_directions)


def calculate_peak_detection_accuracy(y_true: np.ndarray, y_pred: np.ndarray, 
                                    threshold: float = 0.1) -> Dict[str, float]:
    """
    计算峰值检测准确率
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        threshold: 峰值判定阈值
        
    Returns:
        Dict[str, float]: 峰值检测相关指标
    """
    from scipy.signal import find_peaks
    
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # 找到真实峰值
    true_peaks, _ = find_peaks(y_true_flat, height=np.mean(y_true_flat) + threshold * np.std(y_true_flat))
    
    # 找到预测峰值
    pred_peaks, _ = find_peaks(y_pred_flat, height=np.mean(y_pred_flat) + threshold * np.std(y_pred_flat))
    
    if len(true_peaks) == 0:
        return {'peak_precision': 0.0, 'peak_recall': 0.0, 'peak_f1': 0.0}
    
    # 计算峰值匹配 (允许一定的时间偏移)
    tolerance = 3  # 允许3个时间步的偏移
    matched_peaks = 0
    
    for true_peak in true_peaks:
        if any(abs(pred_peak - true_peak) <= tolerance for pred_peak in pred_peaks):
            matched_peaks += 1
    
    precision = matched_peaks / len(pred_peaks) if len(pred_peaks) > 0 else 0.0
    recall = matched_peaks / len(true_peaks)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'peak_precision': precision,
        'peak_recall': recall,
        'peak_f1': f1
    }


def calculate_quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, 
                          quantiles: list = [0.1, 0.5, 0.9]) -> Dict[str, float]:
    """
    计算分位数损失 (用于概率预测评估)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        quantiles: 分位数列表
        
    Returns:
        Dict[str, float]: 各分位数的损失
    """
    losses = {}
    
    for q in quantiles:
        error = y_true - y_pred
        loss = np.mean(np.maximum(q * error, (q - 1) * error))
        losses[f'quantile_loss_{q}'] = float(loss)
    
    return losses


def print_metrics_summary(metrics: Dict[str, float]):
    """
    打印指标摘要
    
    Args:
        metrics: 指标字典
    """
    print("\n" + "="*50)
    print("模型评估指标摘要")
    print("="*50)
    
    print(f"{'指标':<15} {'数值':<15} {'描述'}")
    print("-"*50)
    print(f"{'MSE':<15} {metrics.get('MSE', 0):<15.6f} 均方误差")
    print(f"{'RMSE':<15} {metrics.get('RMSE', 0):<15.6f} 均方根误差")
    print(f"{'MAE':<15} {metrics.get('MAE', 0):<15.6f} 平均绝对误差")
    print(f"{'MAPE (%)':<15} {metrics.get('MAPE', 0):<15.6f} 平均绝对百分比误差")
    print(f"{'R²':<15} {metrics.get('R2', 0):<15.6f} 决定系数")
    print(f"{'RAE':<15} {metrics.get('RAE', 0):<15.6f} 相对绝对误差")
    
    print("\n" + "="*50)
    print("性能评估:")
    if metrics.get('MAPE', 100) < 10:
        print("✅ 优秀 (MAPE < 10%)")
    elif metrics.get('MAPE', 100) < 20:
        print("🟡 良好 (MAPE < 20%)")
    else:
        print("🔴 需要改进 (MAPE >= 20%)")
    print("="*50)