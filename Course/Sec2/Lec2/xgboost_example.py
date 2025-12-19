"""
XGBoost 梯度提升示例

本示例演示如何使用 XGBoost 进行：
1. 回归任务
2. 分类任务
3. 特征重要性分析
"""

import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)

# ========== 1. XGBoost 回归示例 ==========

print("=" * 60)
print("XGBoost 回归示例")
print("=" * 60)

# 生成回归数据
X_reg, y_reg = make_regression(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    noise=10,
    random_state=42
)

# 划分数据集
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# 创建 XGBoost 回归模型
xgb_regressor = xgb.XGBRegressor(
    n_estimators=100,      # 树的数量
    max_depth=3,           # 树的最大深度
    learning_rate=0.1,     # 学习率
    subsample=0.8,         # 每棵树使用的样本比例
    colsample_bytree=0.8,  # 每棵树使用的特征比例
    random_state=42
)

# 训练模型
print("\n训练回归模型...")
xgb_regressor.fit(
    X_train_reg, y_train_reg,
    eval_set=[(X_test_reg, y_test_reg)],
    verbose=False
)

# 预测
y_pred_reg = xgb_regressor.predict(X_test_reg)

# 评估
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"\n回归模型评估:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"R² 分数: {r2:.4f}")

# ========== 2. XGBoost 分类示例 ==========

print("\n" + "=" * 60)
print("XGBoost 分类示例")
print("=" * 60)

# 生成分类数据
X_clf, y_clf = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    random_state=42
)

# 划分数据集
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# 创建 XGBoost 分类模型
xgb_classifier = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'  # 二分类使用对数损失
)

# 训练模型
print("\n训练分类模型...")
xgb_classifier.fit(
    X_train_clf, y_train_clf,
    eval_set=[(X_test_clf, y_test_clf)],
    verbose=False
)

# 预测
y_pred_clf = xgb_classifier.predict(X_test_clf)
y_pred_proba_clf = xgb_classifier.predict_proba(X_test_clf)[:, 1]

# 评估
accuracy = accuracy_score(y_test_clf, y_pred_clf)
print(f"\n分类模型评估:")
print(f"准确率: {accuracy:.4f}")

# 混淆矩阵
cm = confusion_matrix(y_test_clf, y_pred_clf)
print(f"\n混淆矩阵:\n{cm}")

# 分类报告
print(f"\n分类报告:\n{classification_report(y_test_clf, y_pred_clf)}")

# ========== 3. 特征重要性分析 ==========

print("\n" + "=" * 60)
print("特征重要性分析")
print("=" * 60)

# 获取特征重要性
feature_importance_reg = xgb_regressor.feature_importances_
feature_importance_clf = xgb_classifier.feature_importances_

# 创建特征重要性 DataFrame
df_importance_reg = pd.DataFrame({
    'feature': [f'特征_{i+1}' for i in range(len(feature_importance_reg))],
    'importance': feature_importance_reg
}).sort_values('importance', ascending=False)

df_importance_clf = pd.DataFrame({
    'feature': [f'特征_{i+1}' for i in range(len(feature_importance_clf))],
    'importance': feature_importance_clf
}).sort_values('importance', ascending=False)

print("\n回归模型 - 前 5 个重要特征:")
print(df_importance_reg.head())

print("\n分类模型 - 前 5 个重要特征:")
print(df_importance_clf.head())

# ========== 4. 可视化 ==========

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 回归结果
axes[0, 0].scatter(y_test_reg, y_pred_reg, alpha=0.5)
axes[0, 0].plot([y_test_reg.min(), y_test_reg.max()], 
                [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('真实值')
axes[0, 0].set_ylabel('预测值')
axes[0, 0].set_title(f'回归预测结果 (R² = {r2:.4f})')
axes[0, 0].grid(True)

# 回归特征重要性
axes[0, 1].barh(df_importance_reg['feature'], df_importance_reg['importance'])
axes[0, 1].set_xlabel('重要性')
axes[0, 1].set_title('回归模型特征重要性')
axes[0, 1].grid(True, axis='x')

# 分类特征重要性
axes[1, 0].barh(df_importance_clf['feature'], df_importance_clf['importance'])
axes[1, 0].set_xlabel('重要性')
axes[1, 0].set_title('分类模型特征重要性')
axes[1, 0].grid(True, axis='x')

# 学习曲线（使用训练历史）
eval_result_reg = xgb_regressor.evals_result()
axes[1, 1].plot(eval_result_reg['validation_0']['rmse'], label='验证集 RMSE')
axes[1, 1].set_xlabel('迭代次数')
axes[1, 1].set_ylabel('RMSE')
axes[1, 1].set_title('回归模型学习曲线')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('xgboost_results.png', dpi=150)
print("\n结果已保存为 'xgboost_results.png'")

# ========== 5. 交叉验证 ==========

print("\n" + "=" * 60)
print("交叉验证评估")
print("=" * 60)

# 回归交叉验证
cv_scores_reg = cross_val_score(
    xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
    X_reg, y_reg,
    cv=5,
    scoring='neg_mean_squared_error'
)
print(f"回归模型 - 5折交叉验证 RMSE: "
      f"{np.sqrt(-cv_scores_reg.mean()):.4f} (+/- {np.sqrt(cv_scores_reg.std() * 2):.4f})")

# 分类交叉验证
cv_scores_clf = cross_val_score(
    xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
    X_clf, y_clf,
    cv=5,
    scoring='accuracy'
)
print(f"分类模型 - 5折交叉验证准确率: "
      f"{cv_scores_clf.mean():.4f} (+/- {cv_scores_clf.std() * 2):.4f})")

# ========== 6. 超参数调优示例 ==========

print("\n" + "=" * 60)
print("超参数调优示例（网格搜索）")
print("=" * 60)

from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200]
}

# 创建基础模型
xgb_base = xgb.XGBClassifier(random_state=42)

# 网格搜索（使用较小的数据集以加快速度）
print("正在进行网格搜索（这可能需要一些时间）...")
grid_search = GridSearchCV(
    xgb_base,
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_clf[:500], y_train_clf[:500])  # 使用部分数据加快速度

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

# 使用最佳参数训练模型
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_clf)
best_accuracy = accuracy_score(y_test_clf, y_pred_best)
print(f"最佳模型测试准确率: {best_accuracy:.4f}")

# ========== 7. 早停法 (Early Stopping) 示例 ==========

print("\n" + "=" * 60)
print("早停法示例")
print("=" * 60)

xgb_early_stop = xgb.XGBClassifier(
    n_estimators=1000,  # 设置较大的树数量
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)

print("训练模型（使用早停法）...")
xgb_early_stop.fit(
    X_train_clf, y_train_clf,
    eval_set=[(X_test_clf, y_test_clf)],
    early_stopping_rounds=10,  # 如果10轮没有改善就停止
    verbose=False
)

y_pred_early = xgb_early_stop.predict(X_test_clf)
accuracy_early = accuracy_score(y_test_clf, y_pred_early)

print(f"早停法模型 - 实际使用的树数量: {xgb_early_stop.best_iteration}")
print(f"早停法模型 - 测试准确率: {accuracy_early:.4f}")

print("\n所有示例运行完成！")

