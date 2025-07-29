import yaml
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """模型配置"""
    feature_dim: int = 17
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    output_dim: int = 1
    max_seq_length: int = 1000
    dropout: float = 0.1

@dataclass
class DataConfig:
    """數據配置"""
    seq_len: int = 96
    target_len: int = 1
    data_dir: str = "data/processed"
    normalize: bool = True
    scaler_type: str = "minmax"  # minmax, standard
    features: Optional[List[str]] = None
    target_feature: str = "Power_Demand"

@dataclass
class SchedulerConfig:
    """學習率調度器配置"""
    type: str = "ReduceLROnPlateau"
    factor: float = 0.5
    patience: int = 5
    mode: str = "min"

@dataclass
class TrainingConfig:
    """訓練配置"""
    batch_size: int = 32
    learning_rate: float = 1e-4
    train_ratio: float = 0.8
    num_epochs: int = 100
    early_stopping_patience: int = 10
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

@dataclass
class DeviceConfig:
    """設備配置"""
    auto_select: bool = True
    preferred: str = "cuda"

@dataclass
class OutputConfig:
    """輸出配置"""
    save_dir: str = "runs"
    show_plots: bool = False
    save_plots: bool = True

@dataclass
class LoggingConfig:
    """日誌配置"""
    tensorboard: bool = False
    log_dir: str = "logs"
    console_log_level: str = "INFO"

@dataclass
class ClientsConfig:
    """客戶端配置"""
    consumers: Dict[str, Any] = field(default_factory=lambda: {
        'start': 1,
        'end': 50,
        'format': 'Consumer_{:02d}'
    })
    public_building: Dict[str, str] = field(default_factory=lambda: {
        'name': 'Public_Building'
    })

@dataclass
class ExperimentConfig:
    """實驗配置"""
    name: str = "transformer_timeseries"
    version: str = "v1.0"
    description: str = "基於Transformer的時間序列預測模型"

@dataclass
class Config:
    """主配置類"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    clients: ClientsConfig = field(default_factory=ClientsConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

def load_config(config_path: str = "config.yaml") -> Config:
    """
    從YAML文件加載配置
    
    Args:
        config_path: 配置文件路徑
        
    Returns:
        Config: 配置對象
    """
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在，使用默認配置")
        return Config()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)
    
    # 創建配置對象
    config = Config()
    
    # 更新模型配置
    if 'model' in yaml_config:
        for key, value in yaml_config['model'].items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)
    
    # 更新數據配置
    if 'data' in yaml_config:
        for key, value in yaml_config['data'].items():
            if hasattr(config.data, key):
                setattr(config.data, key, value)
    
    # 更新訓練配置
    if 'training' in yaml_config:
        for key, value in yaml_config['training'].items():
            if key == 'scheduler' and isinstance(value, dict):
                # 處理調度器配置
                scheduler_config = SchedulerConfig()
                for sch_key, sch_value in value.items():
                    if hasattr(scheduler_config, sch_key):
                        setattr(scheduler_config, sch_key, sch_value)
                config.training.scheduler = scheduler_config
            elif hasattr(config.training, key):
                setattr(config.training, key, value)
    
    # 更新設備配置
    if 'device' in yaml_config:
        for key, value in yaml_config['device'].items():
            if hasattr(config.device, key):
                setattr(config.device, key, value)
    
    # 更新輸出配置
    if 'output' in yaml_config:
        for key, value in yaml_config['output'].items():
            if hasattr(config.output, key):
                setattr(config.output, key, value)
    
    # 更新日誌配置
    if 'logging' in yaml_config:
        for key, value in yaml_config['logging'].items():
            if hasattr(config.logging, key):
                setattr(config.logging, key, value)
    
    # 更新客戶端配置
    if 'clients' in yaml_config:
        if 'consumers' in yaml_config['clients']:
            config.clients.consumers.update(yaml_config['clients']['consumers'])
        if 'public_building' in yaml_config['clients']:
            config.clients.public_building.update(yaml_config['clients']['public_building'])
    
    # 更新實驗配置
    if 'experiment' in yaml_config:
        for key, value in yaml_config['experiment'].items():
            if hasattr(config.experiment, key):
                setattr(config.experiment, key, value)
    
    return config

def save_config(config: Config, config_path: str = "config.yaml"):
    """
    保存配置到YAML文件
    
    Args:
        config: 配置對象
        config_path: 配置文件路徑
    """
    config_dict = {
        'model': {
            'feature_dim': config.model.feature_dim,
            'd_model': config.model.d_model,
            'nhead': config.model.nhead,
            'num_layers': config.model.num_layers,
            'output_dim': config.model.output_dim,
            'max_seq_length': config.model.max_seq_length,
            'dropout': config.model.dropout
        },
        'data': {
            'seq_len': config.data.seq_len,
            'target_len': config.data.target_len,
            'data_dir': config.data.data_dir,
            'normalize': config.data.normalize,
            'scaler_type': config.data.scaler_type,
            'features': config.data.features,
            'target_feature': config.data.target_feature
        },
        'training': {
            'batch_size': config.training.batch_size,
            'learning_rate': config.training.learning_rate,
            'train_ratio': config.training.train_ratio,
            'num_epochs': config.training.num_epochs,
            'early_stopping_patience': config.training.early_stopping_patience,
            'weight_decay': config.training.weight_decay,
            'grad_clip_norm': config.training.grad_clip_norm,
            'scheduler': {
                'type': config.training.scheduler.type,
                'factor': config.training.scheduler.factor,
                'patience': config.training.scheduler.patience,
                'mode': config.training.scheduler.mode
            }
        },
        'device': {
            'auto_select': config.device.auto_select,
            'preferred': config.device.preferred
        },
        'output': {
            'save_dir': config.output.save_dir,
            'show_plots': config.output.show_plots,
            'save_plots': config.output.save_plots
        },
        'logging': {
            'tensorboard': config.logging.tensorboard,
            'log_dir': config.logging.log_dir,
            'console_log_level': config.logging.console_log_level
        },
        'clients': {
            'consumers': config.clients.consumers,
            'public_building': config.clients.public_building
        },
        'experiment': {
            'name': config.experiment.name,
            'version': config.experiment.version,
            'description': config.experiment.description
        }
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    print(f"配置已保存到 {config_path}")

# 使用示例
if __name__ == "__main__":
    # 加載配置
    config = load_config()
    
    # 打印配置信息
    print("模型配置:")
    print(f"  特徵維度: {config.model.feature_dim}")
    print(f"  隱藏層維度: {config.model.d_model}")
    print(f"  注意力頭數: {config.model.nhead}")
    
    print("\n數據配置:")
    print(f"  序列長度: {config.data.seq_len}")
    print(f"  數據目錄: {config.data.data_dir}")
    print(f"  標準化: {config.data.normalize}")
    
    print("\n訓練配置:")
    print(f"  批次大小: {config.training.batch_size}")
    print(f"  學習率: {config.training.learning_rate}")
    print(f"  訓練輪數: {config.training.num_epochs}")