"""
静态图表绘制
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.dates as mdates
from datetime import datetime, timedelta


# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[Union[str, Path]] = None,
                         figsize: tuple = (12, 5)):
    """
    绘制训练历史曲线
    
    Args:
        history: 包含train_losses和val_losses的字典
        save_path: 保存路径
        figsize: 图像大小
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # 损失曲线
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 学习曲线（平滑版本）
    if len(history['train_losses']) > 10:
        window_size = max(len(history['train_losses']) // 20, 5)
        train_smooth = pd.Series(history['train_losses']).rolling(window_size).mean()
        val_smooth = pd.Series(history['val_losses']).rolling(window_size).mean()
        
        ax2.plot(epochs, train_smooth, 'b-', label=f'Training (smoothed)', linewidth=2)
        ax2.plot(epochs, val_smooth, 'r-', label=f'Validation (smoothed)', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss (Smoothed)')
        ax2.set_title('Smoothed Learning Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图已保存到: {save_path}")
    
    plt.show()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                    save_path: Optional[Union[str, Path]] = None,
                    figsize: tuple = (15, 8),
                    sample_size: int = 200,
                    timestamps: Optional[List] = None):
    """
    绘制预测结果对比图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        save_path: 保存路径
        figsize: 图像大小
        sample_size: 显示的样本数量
        timestamps: 时间戳列表
    """
    # 限制显示的样本数量
    if len(y_true) > sample_size:
        indices = np.linspace(0, len(y_true) - 1, sample_size, dtype=int)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
        if timestamps:
            timestamps_sample = [timestamps[i] for i in indices]
        else:
            timestamps_sample = None
    else:
        y_true_sample = y_true
        y_pred_sample = y_pred
        timestamps_sample = timestamps
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # 时序对比图
    x_axis = timestamps_sample if timestamps_sample else range(len(y_true_sample))
    ax1.plot(x_axis, y_true_sample, 'b-', label='True Values', linewidth=1.5, alpha=0.8)
    ax1.plot(x_axis, y_pred_sample, 'r-', label='Predictions', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.set_title('Predictions vs True Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if timestamps_sample and len(timestamps_sample) > 50:
        ax1.tick_params(axis='x', rotation=45)
    
    # 散点图
    ax2.scatter(y_true_sample, y_pred_sample, alpha=0.6, s=20)
    min_val = min(y_true_sample.min(), y_pred_sample.min())
    max_val = max(y_true_sample.max(), y_pred_sample.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Predictions')
    ax2.set_title('Predictions vs True Values (Scatter Plot)')
    ax2.grid(True, alpha=0.3)
    
    # 残差图
    residuals = y_pred_sample - y_true_sample
    ax3.scatter(y_true_sample, residuals, alpha=0.6, s=20)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('True Values')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Residual Plot')
    ax3.grid(True, alpha=0.3)
    
    # 残差分布
    ax4.hist(residuals, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax4.set_xlabel('Residuals')
    ax4.set_ylabel('Density')
    ax4.set_title('Residual Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测结果图已保存到: {save_path}")
    
    plt.show()


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]],
                          save_path: Optional[Union[str, Path]] = None,
                          figsize: tuple = (12, 8)):
    """
    绘制多个模型的指标对比图
    
    Args:
        metrics_dict: 模型名称到指标字典的映射
        save_path: 保存路径
        figsize: 图像大小
    """
    metrics_df = pd.DataFrame(metrics_dict).T
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    metrics_to_plot = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2', 'RAE']
    
    for i, metric in enumerate(metrics_to_plot):
        if metric in metrics_df.columns:
            ax = axes[i]
            bars = ax.bar(metrics_df.index, metrics_df[metric], alpha=0.7)
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, metrics_df[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"指标对比图已保存到: {save_path}")
    
    plt.show()


def plot_feature_importance(importance_scores: Dict[str, float],
                           save_path: Optional[Union[str, Path]] = None,
                           figsize: tuple = (10, 6)):
    """
    绘制特征重要性图
    
    Args:
        importance_scores: 特征名称到重要性分数的映射
        save_path: 保存路径
        figsize: 图像大小
    """
    features = list(importance_scores.keys())
    scores = list(importance_scores.values())
    
    # 按重要性排序
    sorted_indices = np.argsort(scores)[::-1]
    features_sorted = [features[i] for i in sorted_indices]
    scores_sorted = [scores[i] for i in sorted_indices]
    
    plt.figure(figsize=figsize)
    bars = plt.bar(features_sorted, scores_sorted, alpha=0.7, color='steelblue')
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, score in zip(bars, scores_sorted):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存到: {save_path}")
    
    plt.show()


def plot_attention_weights(attention_weights: np.ndarray,
                          feature_names: List[str],
                          save_path: Optional[Union[str, Path]] = None,
                          figsize: tuple = (12, 8)):
    """
    绘制注意力权重热力图
    
    Args:
        attention_weights: 注意力权重矩阵
        feature_names: 特征名称列表
        save_path: 保存路径
        figsize: 图像大小
    """
    plt.figure(figsize=figsize)
    
    # 创建热力图
    sns.heatmap(attention_weights, 
                xticklabels=feature_names,
                yticklabels=range(attention_weights.shape[0]),
                annot=True, 
                fmt='.3f',
                cmap='Blues',
                cbar_kws={'label': 'Attention Weight'})
    
    plt.title('Attention Weights Heatmap')
    plt.xlabel('Features')
    plt.ylabel('Time Steps')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"注意力权重图已保存到: {save_path}")
    
    plt.show()


def plot_loss_landscape(loss_surface: np.ndarray,
                       param1_range: np.ndarray,
                       param2_range: np.ndarray,
                       param1_name: str = 'Parameter 1',
                       param2_name: str = 'Parameter 2',
                       save_path: Optional[Union[str, Path]] = None,
                       figsize: tuple = (10, 8)):
    """
    绘制损失函数景观图
    
    Args:
        loss_surface: 损失函数表面数据
        param1_range: 参数1的范围
        param2_range: 参数2的范围
        param1_name: 参数1的名称
        param2_name: 参数2的名称
        save_path: 保存路径
        figsize: 图像大小
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(param1_range, param2_range)
    
    surface = ax.plot_surface(X, Y, loss_surface, cmap='viridis', alpha=0.8)
    ax.set_xlabel(param1_name)
    ax.set_ylabel(param2_name)
    ax.set_zlabel('Loss')
    ax.set_title('Loss Landscape')
    
    fig.colorbar(surface, shrink=0.5, aspect=5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"损失景观图已保存到: {save_path}")
    
    plt.show()