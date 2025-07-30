import os
from src.DataLoader import SequenceCSVDataset, create_sequential_datasets
from src.Model import TransformerModel
from src.Trainer import TransformerTrainer
from config import load_config
import requests
from dotenv import load_dotenv

load_dotenv()

def send_message(message):
    if os.getenv('HOST_LINK') is None:
        return
    url = os.getenv('HOST_LINK')
    name = os.getenv('NAME')
    payload = {
        "name": name,
        "message": message
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Error sending message: {e}")

def train(file_path, file_name, config):
    # 創建儲存目錄
    save_dir = os.path.join(config.output.save_dir, file_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 直接從文件計算拆分索引，避免創建臨時數據集
    train_indices, val_indices, test_indices = create_sequential_datasets(
        file_path, 
        config.training.train_ratio,
        config.training.val_ratio,
        config.data.seq_len,
        config.data.target_len
    )
    
    # 創建數據集，傳入訓練索引以確保標準化器僅在訓練集上擬合
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

    # 創建訓練器，傳入預先計算的索引
    trainer = TransformerTrainer(
        model=model, 
        dataset=dataset,
        train_ratio=config.training.train_ratio,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        device=None,  # 使用自動選擇
        show_plot=config.output.show_plots,
        config=config,
        save_dir=save_dir,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices
    )

    # 開始訓練
    trainer.train(
        num_epochs=config.training.num_epochs,
        save_path=os.path.join(save_dir, "transformer_model.pth"),
        early_stopping_patience=config.training.early_stopping_patience,
    )

    # 繪製訓練歷史
    trainer.plot_summary()

def main():
    # 加載配置
    config = load_config()
    
    print("=== 開始批量訓練 ===")
    print(f"實驗名稱: {config.experiment.name}")
    print(f"實驗版本: {config.experiment.version}")
    print(f"實驗描述: {config.experiment.description}")
    print(f"總共需要訓練 51 個模型 (Consumer_01 到 Consumer_50 + Public_Building)")
    print()
    send_message("開始訓練")
    # 訓練消費者模型
    consumers_config = config.clients.consumers
    for i in range(consumers_config['start'], consumers_config['end'] + 1):
        formatted_number = consumers_config['format'].format(i)
        file_path = os.path.join(config.data.data_dir, f"{formatted_number}.csv")
        
        print(f"[{i}/51] 開始訓練 {formatted_number}")
        send_message(f"[{i}/51] 開始訓練 {formatted_number}")
        try:
            train(file_path, formatted_number, config)
            print(f"[{i}/51] {formatted_number} 訓練完成")
        except Exception as e:
            print(f"[{i}/51] {formatted_number} 訓練失敗: {e}")
        print()
    
    # 訓練公共建築模型
    public_building_name = config.clients.public_building['name']
    public_building_path = os.path.join(config.data.data_dir, f"{public_building_name}.csv")
    
    print(f"[51/51] 開始訓練 {public_building_name}")
    send_message(f"[51/51] 開始訓練 {public_building_name}")
    try:
        train(public_building_path, public_building_name, config)
        print(f"[51/51] {public_building_name} 訓練完成")
    except Exception as e:
        print(f"[51/51] {public_building_name} 訓練失敗: {e}")
    print()
    print("=== 所有訓練完成 ===")
    send_message("訓練完成")
if __name__ == "__main__":
    main()