"""
逻辑回归 - 使用 PyTorch 实现

本示例演示如何使用 PyTorch 实现逻辑回归，用于二分类问题。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ========== 1. 生成数据 ==========

# 生成二分类数据集
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42
)

# 转换为 PyTorch 张量
X = torch.FloatTensor(X)
y = torch.FloatTensor(y.reshape(-1, 1))

# 标准化
scaler = StandardScaler()
X_scaled = torch.FloatTensor(scaler.fit_transform(X.numpy()))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")
print(f"类别分布 - 训练集: {y_train.sum().item()} 个正样本, "
      f"{len(y_train) - y_train.sum().item()} 个负样本")

# ========== 2. 定义逻辑回归模型 ==========

class LogisticRegression(nn.Module):
    """逻辑回归模型"""
    
    def __init__(self, input_dim, output_dim=1):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 线性变换 + Sigmoid 激活
        return self.sigmoid(self.linear(x))

# 创建模型
model = LogisticRegression(input_dim=2, output_dim=1)

# 定义损失函数（二元交叉熵）
criterion = nn.BCELoss()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(f"\n模型结构:\n{model}")

# ========== 3. 训练函数 ==========

def train_epoch(model, X_train, y_train, criterion, optimizer):
    """训练一个 epoch"""
    model.train()
    
    # 前向传播
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 计算准确率
    with torch.no_grad():
        y_pred_class = (y_pred > 0.5).float()
        accuracy = (y_pred_class == y_train).float().mean()
    
    return loss.item(), accuracy.item()

# ========== 4. 测试函数 ==========

def evaluate(model, X_test, y_test, criterion):
    """评估模型"""
    model.eval()
    
    with torch.no_grad():
        y_pred = model(X_test)
        loss = criterion(y_pred, y_test)
        
        # 转换为类别
        y_pred_class = (y_pred > 0.5).float()
        accuracy = (y_pred_class == y_test).float().mean()
    
    return loss.item(), accuracy.item(), y_pred_class.numpy()

# ========== 5. 训练模型 ==========

num_epochs = 1000
train_losses = []
train_accs = []
test_losses = []
test_accs = []

print("\n开始训练...")
for epoch in range(num_epochs):
    # 训练
    train_loss, train_acc = train_epoch(model, X_train, y_train, criterion, optimizer)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # 测试
    test_loss, test_acc, _ = evaluate(model, X_test, y_test, criterion)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'训练 Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, '
              f'测试 Loss: {test_loss:.4f}, Acc: {test_acc:.4f}')

# ========== 6. 评估模型 ==========

print("\n=== 模型评估 ===")
test_loss, test_acc, y_pred_class = evaluate(model, X_test, y_test, criterion)
y_test_np = y_test.numpy().flatten()

print(f"测试损失: {test_loss:.4f}")
print(f"测试准确率: {test_acc:.4f}")

# 混淆矩阵
cm = confusion_matrix(y_test_np, y_pred_class.flatten())
print(f"\n混淆矩阵:\n{cm}")

# 分类报告
print(f"\n分类报告:\n{classification_report(y_test_np, y_pred_class.flatten())}")

# ========== 7. 可视化结果 ==========

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 训练曲线
axes[0, 0].plot(train_losses, label='训练损失')
axes[0, 0].plot(test_losses, label='测试损失')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('损失曲线')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(train_accs, label='训练准确率')
axes[0, 1].plot(test_accs, label='测试准确率')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('准确率曲线')
axes[0, 1].legend()
axes[0, 1].grid(True)

# 决策边界可视化
def plot_decision_boundary(model, X, y, ax, title):
    """绘制决策边界"""
    # 创建网格
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 预测网格点
    grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    model.eval()
    with torch.no_grad():
        Z = model(grid_points).numpy()
    Z = Z.reshape(xx.shape)
    
    # 绘制
    ax.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap=plt.cm.RdYlBu)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=plt.cm.RdYlBu, edgecolors='k')
    ax.set_xlabel('特征 1')
    ax.set_ylabel('特征 2')
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax)

# 训练集决策边界
plot_decision_boundary(model, X_train.numpy(), y_train.numpy(), 
                       axes[1, 0], '训练集决策边界')

# 测试集决策边界
plot_decision_boundary(model, X_test.numpy(), y_test.numpy(), 
                       axes[1, 1], '测试集决策边界')

plt.tight_layout()
plt.savefig('logistic_regression_results.png', dpi=150)
print("\n结果已保存为 'logistic_regression_results.png'")

# ========== 8. 多分类逻辑回归示例 ==========

print("\n=== 多分类逻辑回归示例 ===")

# 生成多分类数据
X_multi, y_multi = make_classification(
    n_samples=1000,
    n_features=2,
    n_classes=3,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42
)

X_multi = torch.FloatTensor(X_multi)
y_multi = torch.LongTensor(y_multi)  # 多分类使用 LongTensor

# 标准化
X_multi_scaled = torch.FloatTensor(
    StandardScaler().fit_transform(X_multi.numpy())
)

# 划分数据集
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi_scaled, y_multi, test_size=0.2, random_state=42
)

# 多分类逻辑回归模型（使用 Softmax）
class MultiClassLogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MultiClassLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)  # 使用 CrossEntropyLoss，内部包含 Softmax

# 创建模型
model_multi = MultiClassLogisticRegression(input_dim=2, num_classes=3)
criterion_multi = nn.CrossEntropyLoss()  # 多分类使用交叉熵
optimizer_multi = optim.Adam(model_multi.parameters(), lr=0.01)

# 训练
num_epochs = 1000
for epoch in range(num_epochs):
    model_multi.train()
    y_pred_multi = model_multi(X_train_multi)
    loss = criterion_multi(y_pred_multi, y_train_multi)
    
    optimizer_multi.zero_grad()
    loss.backward()
    optimizer_multi.step()
    
    if (epoch + 1) % 200 == 0:
        model_multi.eval()
        with torch.no_grad():
            y_pred_test = model_multi(X_test_multi)
            _, predicted = torch.max(y_pred_test, 1)
            acc = (predicted == y_test_multi).float().mean()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, '
              f'Acc: {acc.item():.4f}')

# 评估
model_multi.eval()
with torch.no_grad():
    y_pred_test = model_multi(X_test_multi)
    _, predicted = torch.max(y_pred_test, 1)
    acc_multi = (predicted == y_test_multi).float().mean()

print(f"多分类逻辑回归 - 测试准确率: {acc_multi.item():.4f}")

print("\n所有示例运行完成！")

