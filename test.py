import os
import csv
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.DataLoader import SequenceCSVDataset, create_sequential_datasets
from src.Model import TransformerModel
from src.Trainer import TransformerTrainer
from config import load_config

def calculate_metrics(trainer):
    """
    計算模型評估指標
    
    Returns:
        dict: 包含mse, mae, rmse, r2的字典
    """
    trainer.model.eval()
    all_targets = []
    all_predictions = []
    
    # 優先使用測試集，如果沒有測試集則使用驗證集
    data_loader = trainer.test_loader if trainer.test_loader is not None else trainer.val_loader
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.float().to(trainer.device)
            targets = targets.float().to(trainer.device)
            predictions = trainer.model(inputs)
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
    
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    all_predictions = np.concatenate(all_predictions, axis=0).flatten()
    
    # 計算評估指標
    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_predictions)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

def plot_summary(file_path, file_name, config):
    """
    為指定的模型執行plot_summary功能
    
    Args:
        file_path: 數據文件路徑
        file_name: 文件名稱
        config: 配置對象
    """
    # 創建儲存目錄
    save_dir = os.path.join(config.output.save_dir, file_name)
    
    # 檢查模型文件是否存在
    model_path = os.path.join(save_dir, "transformer_model.pth")
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    # 直接從文件計算拆分索引
    train_indices, val_indices, test_indices = create_sequential_datasets(
        file_path, 
        config.training.train_ratio,
        config.training.val_ratio,
        config.data.seq_len,
        config.data.target_len
    )
    
    # 創建數據集
    dataset = SequenceCSVDataset(file_path, config=config, train_indices=train_indices)

    # 創建模型
    model = TransformerModel(
        feature_dim=config.model.feature_dim,
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_layers=config.model.num_layers,
        output_dim=config.model.output_dim,
        max_seq_length=config.model.max_seq_length,
        dropout=config.model.dropout
    )

    # 創建訓練器
    trainer = TransformerTrainer(
        model=model, 
        dataset=dataset,
        train_ratio=config.training.train_ratio,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        device=None,
        show_plot=config.output.show_plots,
        config=config,
        save_dir=save_dir,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices
    )

    # 載入已訓練的模型
    trainer.load_model(model_path)
    
    # 執行plot_summary
    print(f"為 {file_name} 生成圖表...")
    trainer.plot_summary()
    print(f"{file_name} 圖表生成完成")
    
    # 計算並返回評估指標
    metrics = calculate_metrics(trainer)
    return metrics

def main():
    """主函數 - 可以為所有模型或指定模型生成圖表並收集評估結果"""
    config = load_config()
    
    print("=== 開始生成圖表並計算評估指標 ===")
    
    # 儲存所有結果的列表
    results = []
    
    # 為消費者模型生成圖表
    consumers_config = config.clients.consumers
    for i in range(consumers_config['start'], consumers_config['end'] + 1):
        formatted_number = consumers_config['format'].format(i)
        file_path = os.path.join(config.data.data_dir, f"{formatted_number}.csv")
        
        print(f"[{i}/51] 處理 {formatted_number}")
        try:
            metrics = plot_summary(file_path, formatted_number, config)
            if metrics:
                results.append({
                    'client': formatted_number,
                    **metrics
                })
                print(f"MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}, RMSE: {metrics['rmse']:.6f}, R²: {metrics['r2']:.6f}")
        except Exception as e:
            print(f"[{i}/51] {formatted_number} 處理失敗: {e}")
        print()
    
    # 為公共建築模型生成圖表
    public_building_name = config.clients.public_building['name']
    public_building_path = os.path.join(config.data.data_dir, f"{public_building_name}.csv")
    
    print(f"[51/51] 處理 {public_building_name}")
    try:
        metrics = plot_summary(public_building_path, public_building_name, config)
        if metrics:
            results.append({
                'client': public_building_name,
                **metrics
            })
            print(f"MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}, RMSE: {metrics['rmse']:.6f}, R²: {metrics['r2']:.6f}")
    except Exception as e:
        print(f"[51/51] {public_building_name} 處理失敗: {e}")
    
    # 保存結果到CSV文件
    if results:
        csv_path = os.path.join(config.output.save_dir, 'test_results.csv')
        os.makedirs(config.output.save_dir, exist_ok=True)
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['client', 'mse', 'mae', 'rmse', 'r2']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print(f"\n=== 評估結果已保存至 {csv_path} ===")
        print(f"共處理 {len(results)} 個模型")
    else:
        print("\n=== 沒有收集到任何評估結果 ===")
    
    print("=== 所有圖表生成完成 ===")

if __name__ == "__main__":
    main()