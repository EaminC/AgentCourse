# PyTorch 深度学习基础教程

本目录包含 PyTorch 深度学习基础教程和相关示例代码。

## 文件结构

```
Lec2/
├── 课件/
│   └── pytorch_tutorial.md    # 主教程文档
├── mnist_classification.py     # 手写数字识别示例
├── linear_regression.py        # 线性回归示例
├── logistic_regression.py      # 逻辑回归示例
├── xgboost_example.py          # XGBoost 梯度提升示例
└── README.md                   # 本文件
```

## 安装依赖

在运行示例代码之前，请确保安装以下依赖：

```bash
# PyTorch（根据你的系统选择）
# CPU 版本
pip install torch torchvision

# GPU 版本（需要 CUDA）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 其他依赖
pip install numpy matplotlib scikit-learn xgboost pandas
```

## 教程内容

### 1. PyTorch 基础 (`课件/pytorch_tutorial.md`)

- **张量基础**：创建、操作、属性
- **向量运算**：基本运算、矩阵运算、广播机制
- **奇异值分解 (SVD)**：理论、实现、应用（降维、PCA）

### 2. 实战案例

#### 手写数字识别 (`mnist_classification.py`)

使用 PyTorch 构建神经网络识别 MNIST 手写数字。

**运行方式：**

```bash
python mnist_classification.py
```

**主要内容：**

- 数据加载和预处理
- 神经网络模型定义
- 训练和测试流程
- 结果可视化

#### 线性回归 (`linear_regression.py`)

演示三种实现线性回归的方法：

1. 手动实现（使用自动微分）
2. 使用 `nn.Linear`
3. 使用 Adam 优化器

**运行方式：**

```bash
python linear_regression.py
```

**主要内容：**

- 梯度下降算法
- 自动微分机制
- 优化器使用
- 单特征和多特征回归

#### 逻辑回归 (`logistic_regression.py`)

实现二分类和多分类逻辑回归。

**运行方式：**

```bash
python logistic_regression.py
```

**主要内容：**

- 二分类逻辑回归
- 多分类逻辑回归
- 决策边界可视化
- 模型评估指标

#### XGBoost 梯度提升 (`xgboost_example.py`)

使用 XGBoost 进行回归和分类任务。

**运行方式：**

```bash
python xgboost_example.py
```

**主要内容：**

- XGBoost 回归
- XGBoost 分类
- 特征重要性分析
- 交叉验证
- 超参数调优
- 早停法

## 学习路径建议

1. **初学者**：

   - 先阅读 `课件/pytorch_tutorial.md` 了解基础概念
   - 运行 `linear_regression.py` 理解基本流程
   - 运行 `logistic_regression.py` 学习分类问题

2. **进阶**：

   - 运行 `mnist_classification.py` 学习神经网络
   - 运行 `xgboost_example.py` 了解梯度提升

3. **深入**：
   - 修改代码中的超参数，观察效果变化
   - 尝试不同的模型架构
   - 在真实数据集上应用这些方法

## 常见问题

### Q: 如何检查是否安装了 GPU 版本的 PyTorch？

```python
import torch
print(torch.cuda.is_available())  # 如果返回 True，说明可以使用 GPU
```

### Q: 代码运行时出现内存不足错误？

- 减小 `batch_size`
- 减少 `n_estimators`（对于 XGBoost）
- 使用较小的数据集进行测试

### Q: 如何保存和加载模型？

```python
# 保存
torch.save(model.state_dict(), 'model.pth')

# 加载
model = YourModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

## 参考资源

- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [PyTorch 教程](https://pytorch.org/tutorials/)
- [XGBoost 文档](https://xgboost.readthedocs.io/)
- [scikit-learn 文档](https://scikit-learn.org/stable/)

## 注意事项

1. 首次运行 `mnist_classification.py` 时会自动下载 MNIST 数据集
2. 某些示例会生成可视化图片，保存在当前目录
3. 建议在 Jupyter Notebook 中运行代码，便于交互式学习
