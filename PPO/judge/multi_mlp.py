import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import time
import math
import gc

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ========================== 参数边界（保持不变） ==========================
param_bounds = {
    0: (30, 160), 1: (40, 100), 2: (40, 80), 3: (80, 340),
    4: (40, 100), 5: (40, 100), 6: (30, 150), 7: (40, 340),
    8: (40, 100), 9: (30, 100)
}

# ========================== 归一化器（保持不变） ==========================
class ParameterSpecificNormalizer:
    def __init__(self, param_bounds):
        self.param_bounds = param_bounds
        self.num_params = len(param_bounds)
    
    def normalize(self, params):
        params = np.array(params)
        original_shape = params.shape
        if len(original_shape) == 1:
            params = params.reshape(1, -1)
        normalized = np.zeros_like(params, dtype=np.float32)
        for i in range(self.num_params):
            min_val, max_val = self.param_bounds[i]
            normalized[:, i] = (params[:, i] - min_val) / (max_val - min_val)
            normalized[:, i] = np.clip(normalized[:, i], 0, 1)
        return normalized[0] if len(original_shape) == 1 else normalized
    
    def denormalize(self, normalized_params):
        normalized_params = np.array(normalized_params)
        original_shape = normalized_params.shape
        if len(original_shape) == 1:
            normalized_params = normalized_params.reshape(1, -1)
        denormalized = np.zeros_like(normalized_params, dtype=np.float32)
        for i in range(self.num_params):
            min_val, max_val = self.param_bounds[i]
            denormalized[:, i] = normalized_params[:, i] * (max_val - min_val) + min_val
        return denormalized[0] if len(original_shape) == 1 else denormalized

normalizer = ParameterSpecificNormalizer(param_bounds)

# ========================== SwiGLU 激活函数 ==========================
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * torch.nn.functional.silu(gate)

# ========================== MLP 模型（重点替换部分） ==========================
# ========================== 只需要替换模型定义 + 训练部分 ==========================

class EdgeLengthMLPRegressor(nn.Module):
    def __init__(self, input_dim=10, hidden_dims=[256, 512, 512, 256],  # 大幅缩小！
                 dropout=0.4,          # 加大dropout
                 num_outputs=1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h * 2))
            layers.append(nn.LayerNorm(h * 2))
            layers.append(SwiGLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        
        self.backbone = nn.Sequential(*layers)
        self.final_norm = nn.LayerNorm(prev_dim)
        
        self.head = nn.Sequential(
            nn.Dropout(dropout * 0.5),
            nn.Linear(prev_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_outputs)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # === 训练时加输入噪声（非常有效防止过拟合）===
        if self.training:
            x = x + torch.randn_like(x) * 0.05  # 5%的噪声
        
        x = self.backbone(x)
        x = self.final_norm(x)
        x = self.head(x)
        return x.squeeze(-1) if x.size(-1) == 1 else x
# ========================== 数据集（保持不变） ==========================
class EPDataset(Dataset):
    def __init__(self, struct, spectrum, normalizer):
        normalized_struct = normalizer.normalize(struct)
        self.struct = torch.tensor(normalized_struct, dtype=torch.float32)
        self.spectrum = torch.tensor(spectrum, dtype=torch.float32)

    def __getitem__(self, index):
        return self.struct[index], self.spectrum[index]

    def __len__(self):
        return self.struct.shape[0]

# ========================== 数据加载（保持不变） ==========================
def load_data(input_file='pra_reward_dataset.txt'):
    pra_list, reward_list = [], []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if not line or line.startswith('#'): 
                continue
            values = line.split()
            reward = float(values[0])
            pra_data = np.array([float(v) for v in values[1:]], dtype=np.float32)
            pra_list.append(pra_data)
            reward_list.append(reward)
    return np.array(pra_list), np.array(reward_list)

# ========================== 训练函数（基本不变，只改了输出维度处理） ==========================
def train_model(model, dataloaders, criterion, optimizer, scheduler, device, 
                num_epochs=500, writer=None, patience=15):  # patience改小
    best_loss = float('inf')
    best_epoch = 0
    counter = 0  # early stopping计数器
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            preds_list, labels_list = [], []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).view(-1, 1) if labels.ndim == 1 else labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if outputs.ndim == 1:
                        outputs = outputs.unsqueeze(1)
                    
                    # === 可选：Label Smoothing for MSE（对回归也很有用）===
                    # targets = labels * 0.95 + outputs.detach() * 0.05  
                    # loss = criterion(outputs, targets)
                    
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                if phase == 'val':
                    preds_list.append(outputs.detach().cpu())
                    labels_list.append(labels.detach().cpu())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.6f}')

            if writer:
                writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)

            if phase == 'val':
                scheduler.step(epoch_loss)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

                if len(preds_list) > 0:
                    preds = torch.cat(preds_list).numpy().flatten()
                    labels = torch.cat(labels_list).numpy().flatten()
                    mae = np.mean(np.abs(preds - labels))
                    rmse = np.sqrt(np.mean((preds - labels) ** 2))
                    writer.add_scalar('Metrics/MAE', mae, epoch)
                    writer.add_scalar('Metrics/RMSE', rmse, epoch)

                # === Early Stopping 逻辑 ===
                if epoch_loss < best_loss - 1e-5:  # 有效改善才更新
                    best_loss = epoch_loss
                    best_epoch = epoch
                    counter = 0
                    torch.save(model.state_dict(), 'best_mlp_model.pth')
                    print(f"  >>> New best model saved! Val Loss: {best_loss:.6f}")
                else:
                    counter += 1
                    print(f"  No improvement for {counter} epochs (best: {best_loss:.6f} at epoch {best_epoch})")
                    
                    if counter >= patience:
                        print(f"\n=== Early stopping at epoch {epoch+1} ===")
                        print(f"Best Val Loss: {best_loss:.6f} (epoch {best_epoch+1})")
                        return model
    return model
# ========================== 主程序 ==========================
if __name__ == '__main__':
    batch_size = 1024       
    num_epochs = 500
    learning_rate = 1e-3
    weight_decay = 1e-4
    log_dir = f"runs/mlp_experiment_{int(time.time())}"
    writer = SummaryWriter(log_dir=log_dir)

    print("Loading data...")
    pattern_main, reward_main = load_data(input_file='pra_reward_dataset.txt')
    print(f"Loaded {len(pattern_main)} samples")

    x_train, x_val, y_train, y_val = train_test_split(
        pattern_main, reward_main, test_size=0.2, random_state=42, stratify=None)

    image_datasets = {
        'train': EPDataset(x_train, y_train, normalizer),
        'val':   EPDataset(x_val,   y_val,   normalizer)
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True),
        'val':   DataLoader(image_datasets['val'],   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ==== 这里改模型结构 ====
    model = EdgeLengthMLPRegressor(
        input_dim=10,
        hidden_dims=[128, 256, 256, 128],  # 你可以随意增减层数和宽度
        dropout=0.2,    # 128,256,256,128
        num_outputs=1      
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=learning_rate, 
                                  weight_decay=weight_decay)  

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                 patience=8, verbose=True, min_lr=1e-6)  # patience更严格
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = train_model(model, dataloaders, criterion, optimizer, 
                        scheduler, device, num_epochs, writer, patience=15)

    writer.close()
    print(f"\nTensorBoard: tensorboard --logdir=runs")