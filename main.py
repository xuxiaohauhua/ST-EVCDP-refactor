import torch
import argparse
import yaml
import os
from pathlib import Path

from models import TimeSeriesTransformer
from loader import create_dataloaders
from trainer import TransformerTrainer
from metrics import calculate_metrics
from plots import plot_training_history, plot_predictions


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='训练Transformer时序预测模型')
    parser.add_argument('--config', type=str, default='config/transformer_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--data', type=str, required=True,
                       help='CSV数据文件路径')
    parser.add_argument('--features', nargs='+', required=True,
                       help='特征列名')
    parser.add_argument('--target', type=str, required=True,
                       help='目标列名')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader, test_loader, target_scaler = create_dataloaders(
        csv_file=args.data,
        feature_columns=args.features,
        target_column=args.target,
        seq_length=config['model']['seq_length'],
        pred_length=config['model']['pred_length'],
        batch_size=config['training']['batch_size'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio']
    )
    
    print(f"训练集: {len(train_loader)} 批次")
    print(f"验证集: {len(val_loader)} 批次")
    print(f"测试集: {len(test_loader)} 批次")
    
    # 创建模型
    input_dim = len(args.features) + 1  # 特征 + 目标历史值
    model = TimeSeriesTransformer(
        input_dim=input_dim,
        model_dim=config['model']['model_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        seq_length=config['model']['seq_length'],
        pred_length=config['model']['pred_length'],
        dropout=config['model']['dropout']
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    trainer = TransformerTrainer(model, config['training'], device)
    
    # 训练模型
    print("开始训练...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs']
    )
    
    # 测试模型
    print("测试模型...")
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            predictions = model(sequences)
            pred_targets = predictions[:, :, -1].cpu().numpy()
            
            all_predictions.append(pred_targets)
            all_targets.append(targets.numpy())
    
    all_predictions = torch.cat([torch.tensor(p) for p in all_predictions], dim=0)
    all_targets = torch.cat([torch.tensor(t) for t in all_targets], dim=0)
    
    # 反标准化
    pred_original = target_scaler.inverse_transform(all_predictions.reshape(-1, 1)).reshape(all_predictions.shape)
    target_original = target_scaler.inverse_transform(all_targets.reshape(-1, 1)).reshape(all_targets.shape)
    
    # 计算评估指标
    metrics = calculate_metrics(target_original, pred_original)
    print("测试结果:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # 可视化结果
    output_dir = Path('results/transformer')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制训练历史
    plot_training_history(history, save_path=output_dir / 'training_history.png')
    
    # 绘制预测结果
    plot_predictions(target_original[:100], pred_original[:100], 
                    save_path=output_dir / 'predictions.png')
    
    print(f"结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()