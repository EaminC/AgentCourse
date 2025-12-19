# PyTorch 深度学习基础教程

## 目录

1. [PyTorch 简介](#pytorch-简介)
2. [张量基础](#张量基础)
3. [向量运算](#向量运算)
4. [奇异值分解 (SVD)](#奇异值分解-svd)
5. [实战案例](#实战案例)
   - [手写数字识别](#手写数字识别)
   - [线性回归](#线性回归)
   - [逻辑回归](#逻辑回归)
   - [XGBoost 梯度提升](#xgboost-梯度提升)

---

## PyTorch 简介

PyTorch 是一个基于 Python 的深度学习框架，由 Facebook 开发。它提供了：

- **动态计算图**：更灵活的模型构建
- **GPU 加速**：自动利用 CUDA 进行并行计算
- **自动微分**：自动计算梯度
- **丰富的工具库**：数据处理、模型训练等

### 安装

```bash
# CPU 版本
pip install torch torchvision

# GPU 版本（需要 CUDA）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 张量基础

### 什么是张量？

张量（Tensor）是多维数组的推广：

- **0 维张量**：标量（scalar）
- **1 维张量**：向量（vector）
- **2 维张量**：矩阵（matrix）
- **3 维及以上**：高阶张量

### 创建张量

```python
import torch
import numpy as np

# 1. 从列表创建
t1 = torch.tensor([1, 2, 3, 4])
print(f"1维张量: {t1}, shape: {t1.shape}")

# 2. 从 NumPy 数组创建
arr = np.array([[1, 2], [3, 4]])
t2 = torch.from_numpy(arr)
print(f"2维张量:\n{t2}\nshape: {t2.shape}")

# 3. 创建全零张量
zeros = torch.zeros(3, 4)
print(f"全零张量:\n{zeros}")

# 4. 创建全一张量
ones = torch.ones(2, 3)
print(f"全一张量:\n{ones}")

# 5. 创建随机张量
rand = torch.rand(2, 3)  # 均匀分布 [0, 1)
print(f"随机张量:\n{rand}")

# 6. 创建正态分布张量
normal = torch.randn(2, 3)  # 标准正态分布
print(f"正态分布张量:\n{normal}")

# 7. 创建指定范围的张量
arange = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
print(f"范围张量: {arange}")

# 8. 创建指定形状的未初始化张量
empty = torch.empty(2, 3)
print(f"未初始化张量:\n{empty}")
```

### 张量属性

```python
x = torch.randn(3, 4, 5)

print(f"形状 (shape): {x.shape}")
print(f"维度 (dim): {x.dim()}")
print(f"元素总数 (numel): {x.numel()}")
print(f"数据类型 (dtype): {x.dtype}")
print(f"设备 (device): {x.device}")
print(f"是否在 GPU: {x.is_cuda}")
```

### 张量操作

```python
# 1. 重塑张量
x = torch.arange(12)
print(f"原始: {x}, shape: {x.shape}")
x_reshaped = x.reshape(3, 4)
print(f"重塑后:\n{x_reshaped}")

# 2. 转置
matrix = torch.randn(3, 4)
print(f"原始:\n{matrix}")
print(f"转置:\n{matrix.T}")

# 3. 索引和切片
x = torch.arange(12).reshape(3, 4)
print(f"原始:\n{x}")
print(f"第一行: {x[0]}")
print(f"第一列: {x[:, 0]}")
print(f"子矩阵:\n{x[0:2, 1:3]}")

# 4. 连接张量
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
print(f"垂直连接:\n{torch.cat([a, b], dim=0)}")
print(f"水平连接:\n{torch.cat([a, b], dim=1)}")

# 5. 堆叠张量
print(f"堆叠:\n{torch.stack([a, b], dim=0)}")
```

---

## 向量运算

### 基本运算

```python
# 创建向量
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# 1. 加法
print(f"a + b = {a + b}")

# 2. 减法
print(f"a - b = {a - b}")

# 3. 标量乘法
print(f"2 * a = {2 * a}")

# 4. 逐元素乘法
print(f"a * b = {a * b}")

# 5. 点积（内积）
print(f"a · b = {torch.dot(a, b)}")

# 6. 向量范数
print(f"||a||₂ = {torch.norm(a)}")  # L2 范数
print(f"||a||₁ = {torch.abs(a).sum()}")  # L1 范数
```

### 矩阵运算

```python
# 创建矩阵
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

# 1. 矩阵乘法
print(f"A @ B =\n{A @ B}")

# 2. 逐元素运算
print(f"A * B =\n{A * B}")  # 逐元素乘法
print(f"A ** 2 =\n{A ** 2}")  # 逐元素平方

# 3. 矩阵转置
print(f"A^T =\n{A.T}")

# 4. 矩阵求逆
A_inv = torch.inverse(A)
print(f"A^(-1) =\n{A_inv}")
print(f"A @ A^(-1) =\n{A @ A_inv}")  # 验证

# 5. 矩阵的行列式
det = torch.det(A)
print(f"det(A) = {det}")

# 6. 矩阵的迹
trace = torch.trace(A)
print(f"tr(A) = {trace}")
```

### 广播机制

```python
# 广播允许不同形状的张量进行运算
a = torch.arange(3).reshape(3, 1)  # shape: (3, 1)
b = torch.arange(2)  # shape: (2,)

print(f"a:\n{a}")
print(f"b: {b}")
print(f"a + b:\n{a + b}")  # 自动广播到 (3, 2)
```

---

## 奇异值分解 (SVD)

### SVD 理论

奇异值分解（Singular Value Decomposition）将矩阵分解为三个矩阵的乘积：

**A = U Σ V^T**

其中：

- **U**：左奇异向量矩阵（m × m）
- **Σ**：奇异值对角矩阵（m × n）
- **V^T**：右奇异向量矩阵的转置（n × n）

### SVD 在 PyTorch 中的实现

```python
import torch
import matplotlib.pyplot as plt

# 创建一个示例矩阵
A = torch.randn(5, 3)
print(f"原始矩阵 A (5×3):\n{A}\n")

# 执行 SVD
U, S, V = torch.linalg.svd(A, full_matrices=False)

print(f"U 矩阵 (5×3):\n{U}\n")
print(f"奇异值 S: {S}")
print(f"V 矩阵 (3×3):\n{V}\n")

# 重构矩阵
Sigma = torch.diag(S)
A_reconstructed = U @ Sigma @ V.T
print(f"重构的 A:\n{A_reconstructed}\n")
print(f"重构误差: {torch.norm(A - A_reconstructed)}")

# 验证 U 和 V 的正交性
print(f"U^T @ U 是否为单位矩阵: {torch.allclose(U.T @ U, torch.eye(3))}")
print(f"V^T @ V 是否为单位矩阵: {torch.allclose(V.T @ V, torch.eye(3))}")
```

### SVD 应用：降维

```python
# 使用 SVD 进行降维
def svd_compress(A, k):
    """
    使用 SVD 将矩阵压缩到 k 维

    参数:
        A: 输入矩阵 (m, n)
        k: 保留的维度数

    返回:
        压缩后的矩阵
    """
    U, S, V = torch.linalg.svd(A, full_matrices=False)

    # 只保留前 k 个奇异值
    U_k = U[:, :k]
    S_k = S[:k]
    V_k = V[:k, :]

    # 重构
    A_compressed = U_k @ torch.diag(S_k) @ V_k

    return A_compressed, S

# 示例：压缩图像矩阵
# 创建一个模拟的 100×100 图像矩阵
image = torch.randn(100, 100)

# 压缩到 10 维
compressed, singular_values = svd_compress(image, k=10)

# 计算压缩比
original_size = image.numel()
compressed_size = 10 * (100 + 100 + 1)  # U_k, V_k, S_k 的元素数
compression_ratio = compressed_size / original_size

print(f"原始大小: {original_size}")
print(f"压缩后大小: {compressed_size}")
print(f"压缩比: {compression_ratio:.4f}")
print(f"重构误差: {torch.norm(image - compressed):.4f}")

# 可视化奇异值
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(singular_values[:20].numpy())
plt.title('前 20 个奇异值')
plt.xlabel('索引')
plt.ylabel('奇异值')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.semilogy(singular_values.numpy())
plt.title('所有奇异值（对数尺度）')
plt.xlabel('索引')
plt.ylabel('奇异值（对数）')
plt.grid(True)
plt.tight_layout()
plt.show()
```

### SVD 应用：主成分分析 (PCA)

```python
def pca_svd(X, n_components=2):
    """
    使用 SVD 实现 PCA

    参数:
        X: 数据矩阵 (n_samples, n_features)
        n_components: 主成分数量

    返回:
        降维后的数据
    """
    # 中心化数据
    X_centered = X - X.mean(dim=0)

    # SVD
    U, S, V = torch.linalg.svd(X_centered, full_matrices=False)

    # 主成分（前 n_components 个右奇异向量）
    components = V[:n_components, :].T

    # 投影到主成分空间
    X_reduced = X_centered @ components

    return X_reduced, components, S

# 示例：对 2D 数据降维
# 生成示例数据
torch.manual_seed(42)
X = torch.randn(100, 10)

# 降维到 2D
X_reduced, components, singular_values = pca_svd(X, n_components=2)

print(f"原始数据形状: {X.shape}")
print(f"降维后形状: {X_reduced.shape}")
print(f"主成分解释的方差比例: {(singular_values[:2]**2 / (singular_values**2).sum()).item():.4f}")
```

---

## 实战案例

### 手写数字识别

详见 `mnist_classification.py`

### 线性回归

详见 `linear_regression.py`

### 逻辑回归

详见 `logistic_regression.py`

### XGBoost 梯度提升

详见 `xgboost_example.py`

---

## 总结

本教程涵盖了：

1. **张量基础**：创建、操作、属性
2. **向量运算**：基本运算、矩阵运算、广播
3. **SVD**：理论、实现、应用（降维、PCA）
4. **实战案例**：四个完整的机器学习项目

### 下一步学习

- 深度学习：神经网络、CNN、RNN
- 优化算法：梯度下降、Adam、学习率调度
- 正则化：Dropout、Batch Normalization
- 迁移学习：预训练模型的使用

### 参考资源

- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [PyTorch 教程](https://pytorch.org/tutorials/)
- [深度学习花书](https://www.deeplearningbook.org/)
