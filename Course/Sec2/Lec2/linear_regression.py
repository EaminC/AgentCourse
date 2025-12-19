"""
线性回归 - 使用 PyTorch 实现

本示例演示如何使用 PyTorch 实现线性回归，包括：
1. 手动实现梯度下降
2. 使用 PyTorch 的自动微分
3. 使用优化器
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ========== 1. 生成数据 ==========

# 生成回归数据集
X, y = make_regression(
    n_samples=1000,
    n_features=1,
    noise=10,
    random_state=42
)

# 转换为 PyTorch 张量
X = torch.FloatTensor(X)
y = torch.FloatTensor(y.reshape(-1, 1))

# 标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = torch.FloatTensor(scaler_X.fit_transform(X.numpy()))
y_scaled = torch.FloatTensor(scaler_y.fit_transform(y.numpy()))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# ========== 2. 方法一：手动实现线性回归 ==========

print("\n=== 方法一：手动实现（使用自动微分）===")

# 初始化参数
w = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, 1, requires_grad=True)

learning_rate = 0.01
num_epochs = 1000
losses_manual = []

for epoch in range(num_epochs):
    # 前向传播
    y_pred = X_train @ w + b
    
    # 计算损失（均方误差）
    loss = torch.mean((y_pred - y_train) ** 2)
    
    # 反向传播
    loss.backward()
    
    # 更新参数（手动梯度下降）
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        
        # 清零梯度
        w.grad.zero_()
        b.grad.zero_()
    
    losses_manual.append(loss.item())
    
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试
with torch.no_grad():
    y_pred_manual = X_test @ w + b
    test_loss_manual = torch.mean((y_pred_manual - y_test) ** 2).item()

print(f"手动实现 - 测试损失: {test_loss_manual:.4f}")
print(f"权重 w: {w.item():.4f}, 偏置 b: {b.item():.4f}")

# ========== 3. 方法二：使用 PyTorch 的 nn.Linear ==========

print("\n=== 方法二：使用 PyTorch 的 nn.Linear ===")

# 定义模型
class LinearRegression(nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

# 创建模型
model = LinearRegression()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练
num_epochs = 1000
losses_nn = []

for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses_nn.append(loss.item())
    
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试
model.eval()
with torch.no_grad():
    y_pred_nn = model(X_test)
    test_loss_nn = criterion(y_pred_nn, y_test).item()

print(f"PyTorch nn - 测试损失: {test_loss_nn:.4f}")
print(f"权重 w: {model.linear.weight.item():.4f}, "
      f"偏置 b: {model.linear.bias.item():.4f}")

# ========== 4. 方法三：使用 Adam 优化器 ==========

print("\n=== 方法三：使用 Adam 优化器 ===")

model_adam = LinearRegression()
criterion = nn.MSELoss()
optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.01)

num_epochs = 500
losses_adam = []

for epoch in range(num_epochs):
    y_pred = model_adam(X_train)
    loss = criterion(y_pred, y_train)
    
    optimizer_adam.zero_grad()
    loss.backward()
    optimizer_adam.step()
    
    losses_adam.append(loss.item())
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试
model_adam.eval()
with torch.no_grad():
    y_pred_adam = model_adam(X_test)
    test_loss_adam = criterion(y_pred_adam, y_test).item()

print(f"Adam 优化器 - 测试损失: {test_loss_adam:.4f}")

# ========== 5. 可视化结果 ==========

# 反标准化预测结果
y_pred_manual_orig = scaler_y.inverse_transform(y_pred_manual.numpy())
y_pred_nn_orig = scaler_y.inverse_transform(y_pred_nn.numpy())
y_pred_adam_orig = scaler_y.inverse_transform(y_pred_adam.numpy())
y_test_orig = scaler_y.inverse_transform(y_test.numpy())
X_test_orig = scaler_X.inverse_transform(X_test.numpy())

# 绘制结果
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 损失曲线
axes[0, 0].plot(losses_manual, label='手动实现')
axes[0, 0].plot(losses_nn, label='nn.Linear (SGD)')
axes[0, 0].plot(losses_adam, label='nn.Linear (Adam)')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('训练损失曲线')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 手动实现的结果
axes[0, 1].scatter(X_test_orig, y_test_orig, alpha=0.5, label='真实值')
axes[0, 1].scatter(X_test_orig, y_pred_manual_orig, alpha=0.5, label='预测值')
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('y')
axes[0, 1].set_title(f'手动实现 (Loss: {test_loss_manual:.4f})')
axes[0, 1].legend()
axes[0, 1].grid(True)

# nn.Linear 的结果
axes[1, 0].scatter(X_test_orig, y_test_orig, alpha=0.5, label='真实值')
axes[1, 0].scatter(X_test_orig, y_pred_nn_orig, alpha=0.5, label='预测值')
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('y')
axes[1, 0].set_title(f'nn.Linear + SGD (Loss: {test_loss_nn:.4f})')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Adam 的结果
axes[1, 1].scatter(X_test_orig, y_test_orig, alpha=0.5, label='真实值')
axes[1, 1].scatter(X_test_orig, y_pred_adam_orig, alpha=0.5, label='预测值')
axes[1, 1].set_xlabel('X')
axes[1, 1].set_ylabel('y')
axes[1, 1].set_title(f'nn.Linear + Adam (Loss: {test_loss_adam:.4f})')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('linear_regression_results.png', dpi=150)
print("\n结果已保存为 'linear_regression_results.png'")

# ========== 6. 多特征线性回归示例 ==========

print("\n=== 多特征线性回归示例 ===")

# 生成多特征数据
X_multi, y_multi = make_regression(
    n_samples=1000,
    n_features=5,
    noise=10,
    random_state=42
)

X_multi = torch.FloatTensor(X_multi)
y_multi = torch.FloatTensor(y_multi.reshape(-1, 1))

# 标准化
X_multi_scaled = torch.FloatTensor(
    StandardScaler().fit_transform(X_multi.numpy())
)
y_multi_scaled = torch.FloatTensor(
    StandardScaler().fit_transform(y_multi.numpy())
)

# 划分数据集
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi_scaled, y_multi_scaled, test_size=0.2, random_state=42
)

# 创建模型
model_multi = LinearRegression(input_dim=5, output_dim=1)
criterion = nn.MSELoss()
optimizer_multi = optim.Adam(model_multi.parameters(), lr=0.01)

# 训练
for epoch in range(500):
    y_pred_multi = model_multi(X_train_multi)
    loss = criterion(y_pred_multi, y_train_multi)
    
    optimizer_multi.zero_grad()
    loss.backward()
    optimizer_multi.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/500], Loss: {loss.item():.4f}')

# 测试
model_multi.eval()
with torch.no_grad():
    y_pred_multi_test = model_multi(X_test_multi)
    test_loss_multi = criterion(y_pred_multi_test, y_test_multi).item()

print(f"多特征线性回归 - 测试损失: {test_loss_multi:.4f}")
print(f"权重: {model_multi.linear.weight.squeeze().tolist()}")
print(f"偏置: {model_multi.linear.bias.item():.4f}")

print("\n所有示例运行完成！")

