"""
手写数字识别 - 使用 PyTorch 和 MNIST 数据集

本示例演示如何使用 PyTorch 构建一个简单的神经网络来识别手写数字。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# ========== 1. 数据准备 ==========

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将 PIL 图像转换为张量
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值和标准差
])

# 加载训练集和测试集
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2
)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# ========== 2. 定义模型 ==========

class SimpleNN(nn.Module):
    """简单的全连接神经网络"""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # 展平输入 (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.view(x.size(0), -1)
        
        # 前向传播
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

# 创建模型
model = SimpleNN(input_size=784, hidden_size=128, num_classes=10)
print(f"\n模型结构:\n{model}")

# 计算模型参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数数量: {total_params:,}")

# ========== 3. 定义损失函数和优化器 ==========

criterion = nn.CrossEntropyLoss()  # 交叉熵损失（适用于多分类）
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

# ========== 4. 训练函数 ==========

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if (batch_idx + 1) % 200 == 0:
            print(f'  Batch [{batch_idx+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ========== 5. 测试函数 ==========

def test(model, test_loader, criterion, device):
    """测试模型"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc

# ========== 6. 训练模型 ==========

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用设备: {device}")
model = model.to(device)

num_epochs = 5
train_losses = []
train_accs = []
test_losses = []
test_accs = []

print("\n开始训练...")
for epoch in range(num_epochs):
    print(f'\nEpoch [{epoch+1}/{num_epochs}]')
    
    # 训练
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # 测试
    test_loss, test_acc = test(model, test_loader, criterion, device)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    print(f'训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
    print(f'测试 - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')

# ========== 7. 可视化结果 ==========

# 绘制训练曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='训练损失')
plt.plot(test_losses, label='测试损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('损失曲线')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='训练准确率')
plt.plot(test_accs, label='测试准确率')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('准确率曲线')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('mnist_training_curves.png', dpi=150)
print("\n训练曲线已保存为 'mnist_training_curves.png'")

# ========== 8. 可视化一些预测结果 ==========

def visualize_predictions(model, test_loader, device, num_samples=8):
    """可视化预测结果"""
    model.eval()
    
    # 获取一批测试数据
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # 可视化
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        img = images[i].cpu().squeeze()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'真实: {labels[i].item()}, 预测: {predicted[i].item()}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_predictions.png', dpi=150)
    print("预测结果已保存为 'mnist_predictions.png'")

visualize_predictions(model, test_loader, device)

print("\n训练完成！")
print(f"最终测试准确率: {test_accs[-1]:.2f}%")

